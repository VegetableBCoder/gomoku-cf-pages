/** 15×15 五子棋棋盘逻辑，与 Python 侧 gomoku/board.py 行为对齐。
 *
 * 规则: 自由规则(Freestyle) —— 连五及以上(长连)即胜。
 */

export const EMPTY = 0;
export const BLACK = 1;
export const WHITE = 2;

export class Board {
  constructor(size = 15) {
    this.size = size;
    this.grid = new Uint8Array(size * size);
    this.history = [];            // [{ r, c, color }]
  }

  inBounds(r, c) {
    return r >= 0 && r < this.size && c >= 0 && c < this.size;
  }

  get(r, c) {
    return this.grid[r * this.size + c];
  }

  isEmpty(r, c) {
    return this.inBounds(r, c) && this.grid[r * this.size + c] === EMPTY;
  }

  isFull() {
    return this.history.length === this.size * this.size;
  }

  place(r, c, color) {
    if (!this.isEmpty(r, c)) return false;
    this.grid[r * this.size + c] = color;
    this.history.push({ r, c, color });
    return true;
  }

  undo() {
    const last = this.history.pop();
    if (!last) return false;
    this.grid[last.r * this.size + last.c] = EMPTY;
    return true;
  }

  /** 返回胜方颜色 (BLACK/WHITE)，无胜方返回 EMPTY。长连(>=5)算胜。 */
  winner() {
    const n = this.size;
    const g = this.grid;
    const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const color = g[r * n + c];
        if (color === EMPTY) continue;
        for (const [dr, dc] of dirs) {
          const pr = r - dr;
          const pc = c - dc;
          if (this.inBounds(pr, pc) && g[pr * n + pc] === color) continue;
          let cnt = 0;
          let rr = r;
          let cc = c;
          while (this.inBounds(rr, cc) && g[rr * n + cc] === color) {
            cnt++;
            rr += dr;
            cc += dc;
          }
          if (cnt >= 5) return color;
        }
      }
    }
    return EMPTY;
  }

  /** 落子颜色: 黑先手，按 history 长度交替。 */
  sideToMove() {
    return this.history.length % 2 === 0 ? BLACK : WHITE;
  }
}
