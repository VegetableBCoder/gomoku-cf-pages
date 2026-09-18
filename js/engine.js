/** ONNX 推理封装：与 Python 侧 gomoku/players/nn_player.py 的编码/选点行为对齐。
 *
 * 输入 (1,2,15,15) NCHW, 通道0=自己 通道1=对手, 值 0/1。
 * 输出 policy_logits (1,225) 未过 softmax, value_logits (1,3) = [胜, 负, 和]。
 *
 * fp16 模型的输入用 Uint16Array 位模式构造（ORT 任何版本都接受并自行转换，
 * 比用 Float16Array 更兼容旧浏览器）；输出的解码见 decode() 的注释。
 */
import * as ort from 'onnxruntime-web/webgpu';
import { candidates, pickMove } from './policy.js';

export const BOARD_N = 15;

const F16_ZERO = 0x0000;
const F16_ONE = 0x3c00;          // 1.0 在 fp16 下的位表示（精确）

export class NNEngine {
  constructor(session, providers) {
    const meta = session.inputMetadata?.[0];
    this.session = session;
    this.providers = providers;
    this.inputName = (meta && meta.name) || session.inputNames[0];
    this.float16 = /float16/.test((meta && meta.type) || '');
  }

  static async load(url, { providers = ['webgpu'] } = {}) {
    const session = await ort.InferenceSession.create(url, {
      executionProviders: providers,
      graphOptimizationLevel: 'all',
    });
    return new NNEngine(session, await resolveProviders(providers));
  }

  /** 通道0=自己, 通道1=对手。fp16 模型时直接按位写 1.0。 */
  encode(board, selfColor) {
    const n = board.size;
    const size = 2 * n * n;
    if (this.float16) {
      const planes = new Uint16Array(size).fill(F16_ZERO);
      for (const { r, c, color } of board.history) {
        const ch = color === selfColor ? 0 : 1;
        planes[ch * n * n + r * n + c] = F16_ONE;
      }
      return planes;
    }
    const planes = new Float32Array(size);
    for (const { r, c, color } of board.history) {
      const ch = color === selfColor ? 0 : 1;
      planes[ch * n * n + r * n + c] = 1;
    }
    return planes;
  }

  /** 跑一次前向，返回 { policy: Float32Array(225), value: Float32Array(3) }。 */
  async infer(board, selfColor) {
    const n = board.size;
    const data = this.encode(board, selfColor);
    const tensor = this.float16
      ? new ort.Tensor('float16', data, [1, 2, n, n])
      : new ort.Tensor('float32', data, [1, 2, n, n]);
    const out = await this.session.run({ [this.inputName]: tensor });
    return {
      policy: decode(out.policy_logits),
      value: decode(out.value_logits),
    };
  }

  /** 选点 + 顺手返回 value 头。value 是该局面下 selfColor 的 [胜, 负, 和] logits。 */
  async chooseMove(board, selfColor, { temperature = 0, radius = 2 } = {}) {
    const cands = candidates(board, radius);
    if (cands.length === 0) return { move: null, value: null };
    const { policy, value } = await this.infer(board, selfColor);
    return { move: pickMove(policy, cands, { temperature }), value };
  }
}

/** 探测给定 provider 列表中实际生效的那些。
 *
 * ORT 在 EP 不可用时会静默降级（只在 console 打一行 warning），
 * 若直接展示请求列表就会谎报「跑在 webgpu」而实际是 wasm。
 * 注: 这只是环境探测（能否拿到 adapter），不等于 ORT 的 WebGPU kernel 必然初始化成功；
 *     若 webgpu 后端自身初始化失败，ORT 仍会降级到 wasm，此时显示会偏乐观。
 */
async function resolveProviders(requested) {
  const out = [];
  for (const p of requested) {
    if (p === 'webgpu') {
      // WebGPU 可用性取决于运行时（无头浏览器常拿不到 adapter）
      if (typeof navigator !== 'undefined' && navigator.gpu) {
        try {
          if (await navigator.gpu.requestAdapter()) out.push(p);
        } catch { /* 拿不到 adapter，视为不可用 */ }
      }
    } else {
      out.push(p);      // wasm 始终可用（CPU 兜底）
    }
  }
  // 一个都不剩时，ORT 实际会跑 wasm 兜底，如实报告而不是回退成请求值
  return out.length ? out : ['wasm'];
}

/** 输出张量 → Float32Array。
 *
 * 注意 fp16 有两种数据表示，必须按 tensor.data 的实际类型分派，不能按 tensor.type 猜：
 *   Float16Array  ORT 1.30 + 支持原生 Float16Array 的浏览器（Chrome 135+）→ 元素已是数值
 *                 （即使调用方用 Uint16Array 构造，ORT 内部也存成 Float16Array）
 *   Uint16Array   旧版 ORT / 无 Float16Array 的环境 → 元素是 IEEE-754 binary16 原始位
 * 曾按 type 无条件调 f16ToF32，把已有数值当位模式再解一次，导致 NaN→0 与整盘 logits 报废。
 */
function decode(tensor) {
  const d = tensor.data;
  if (d instanceof Float32Array) return d;
  if (tensor.type !== 'float16') return Float32Array.from(d);
  if (typeof Float16Array !== 'undefined' && d instanceof Float16Array) {
    return Float32Array.from(d);              // 已经是数值，直接转宽
  }
  if (d instanceof Uint16Array) {             // 位模式，需手工解码
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = f16ToF32(d[i]);
    return out;
  }
  throw new TypeError(
    `无法识别的 float16 张量数据: ${d?.constructor?.name ?? typeof d}`);
}

function f16ToF32(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

/** value_logits → [胜, 负, 和] 概率。 */
export function valueProbs(value) {
  const mx = Math.max(...value);
  const exps = Array.from(value, (v) => Math.exp(v - mx));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}
