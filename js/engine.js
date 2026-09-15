/** ONNX 推理封装：与 Python 侧 gomoku/players/nn_player.py 的编码/选点行为对齐。
 *
 * 输入 (1,2,15,15) NCHW, 通道0=自己 通道1=对手, 值 0/1。
 * 输出 policy_logits (1,225) 未过 softmax, value_logits (1,3) = [胜, 负, 和]。
 *
 * 注: ORT Web 的 fp16 张量是 Uint16Array 里的原始半精度位, 不是 fp32 数组,
 *     所以输入要按位构造、输出要按位解码。
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
    return new NNEngine(session, providers);
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

/** 输出张量 → Float32Array（fp16 时按位解码）。 */
function decode(tensor) {
  const d = tensor.data;
  if (d instanceof Float32Array) return d;
  if (tensor.type !== 'float16') return Float32Array.from(d);
  const out = new Float32Array(d.length);
  for (let i = 0; i < d.length; i++) out[i] = f16ToF32(d[i]);
  return out;
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
