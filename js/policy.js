/** 选点逻辑（纯函数，不依赖 onnxruntime），与 nn_player.py 对齐。 */

export const HW = 7;               // (15-1)/2

/** 以已有棋子为中心 radius 邻域内的空点；空盘返回中心点。候选按 (r,c) 升序。 */
export function candidates(board, radius = 2) {
  const n = board.size;
  if (board.history.length === 0) return [[(n - 1) >> 1, (n - 1) >> 1]];
  const seen = new Uint8Array(n * n);
  const out = [];
  for (const { r, c } of board.history) {
    for (let dr = -radius; dr <= radius; dr++) {
      for (let dc = -radius; dc <= radius; dc++) {
        const rr = r + dr;
        const cc = c + dc;
        if (!board.inBounds(rr, cc)) continue;
        const idx = rr * n + cc;
        if (seen[idx] || board.grid[idx] !== 0) continue;
        seen[idx] = 1;
        out.push([rr, cc]);
      }
    }
  }
  out.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  return out;
}

/** temperature=0 贪心取最大 logit（并列取最靠中心）；>0 按 softmax 采样。 */
export function pickMove(policy, cands, { temperature = 0, rng = Math.random } = {}) {
  if (cands.length === 0) return null;
  const n = 15;

  if (temperature > 0) {
    const t = Math.max(temperature, 1e-3);
    const logits = cands.map(([r, c]) => policy[r * n + c]);
    const mx = Math.max(...logits);
    const exps = logits.map((v) => Math.exp((v - mx) / t));
    const sum = exps.reduce((a, b) => a + b, 0);
    let pick = rng();
    for (let i = 0; i < cands.length; i++) {
      pick -= exps[i] / sum;
      if (pick <= 0) return cands[i];
    }
    return cands[cands.length - 1];
  }

  let best = cands[0];
  let bestLogit = -Infinity;
  let bestDist = Infinity;
  for (const [r, c] of cands) {
    const v = policy[r * n + c];
    const d = (r - HW) ** 2 + (c - HW) ** 2;
    if (v > bestLogit || (v === bestLogit && d < bestDist)) {
      bestLogit = v;
      bestDist = d;
      best = [r, c];
    }
  }
  return best;
}
