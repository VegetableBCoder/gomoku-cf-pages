/** 页面交互：棋盘渲染 + 人机对局。 */
import { Board, BLACK, WHITE, EMPTY } from './board.js';
import { NNEngine, valueProbs, BOARD_N } from './engine.js';

const N = BOARD_N;
const CELL = 36;
const MARGIN = 28;
const CANVAS = MARGIN * 2 + (N - 1) * CELL;
const R_STONE = CELL * 0.42;
const STARS = [[3, 3], [3, 7], [3, 11], [7, 3], [7, 7], [7, 11], [11, 3], [11, 7], [11, 11]];

// 只提供 fp16：fp32 是 30.5 MiB，超过 Cloudflare Pages 单文件 25 MiB 上限。
// 权重不入库，托管在 Cloudflare R2，跨域拉取需要 bucket 配好 CORS。
const MODELS = {
  fp16: 'https://pub-b385817d3bb84eac83e547468ba1b208.r2.dev/gomoku/models/gomoku_fp16.onnx',
};

const state = {
  board: new Board(N),
  humanColor: BLACK,
  engine: null,
  busy: false,
  gameOver: false,
  lastMove: null,
  evalText: '',
  hover: null,
  pending: false,
};

const $ = (id) => document.getElementById(id);
const canvas = $('board');
const ctx = canvas.getContext('2d');

function setupCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = CANVAS * dpr;
  canvas.height = CANVAS * dpr;
  canvas.style.width = `${CANVAS}px`;
  canvas.style.height = `${CANVAS}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

const px = (i) => MARGIN + i * CELL;

function drawBoard() {
  ctx.clearRect(0, 0, CANVAS, CANVAS);

  ctx.fillStyle = '#dcb35c';
  ctx.fillRect(0, 0, CANVAS, CANVAS);

  ctx.strokeStyle = '#7a5c22';
  ctx.lineWidth = 1;
  for (let i = 0; i < N; i++) {
    ctx.beginPath();
    ctx.moveTo(px(0), px(i));
    ctx.lineTo(px(N - 1), px(i));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(px(i), px(0));
    ctx.lineTo(px(i), px(N - 1));
    ctx.stroke();
  }

  ctx.fillStyle = '#4a3712';
  for (const [r, c] of STARS) {
    ctx.beginPath();
    ctx.arc(px(c), px(r), 3.2, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawStone(r, c, color) {
  const x = px(c);
  const y = px(r);
  const g = ctx.createRadialGradient(x - R_STONE * 0.35, y - R_STONE * 0.35,
    R_STONE * 0.1, x, y, R_STONE);
  if (color === BLACK) {
    g.addColorStop(0, '#6b6b6b');
    g.addColorStop(1, '#0a0a0a');
  } else {
    g.addColorStop(0, '#ffffff');
    g.addColorStop(1, '#c9c9c9');
  }
  ctx.beginPath();
  ctx.arc(x, y, R_STONE, 0, Math.PI * 2);
  ctx.fillStyle = g;
  ctx.fill();
  ctx.strokeStyle = color === BLACK ? '#000' : '#9a9a9a';
  ctx.lineWidth = 0.8;
  ctx.stroke();
}

function drawLastMove(r, c) {
  ctx.beginPath();
  ctx.arc(px(c), px(r), R_STONE * 0.35, 0, Math.PI * 2);
  ctx.fillStyle = '#e23b3b';
  ctx.fill();
}

function drawHover(r, c) {
  if (!state.board.isEmpty(r, c)) return;
  ctx.beginPath();
  ctx.arc(px(c), px(r), R_STONE, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(40,90,180,.55)';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function render() {
  drawBoard();
  for (const { r, c, color } of state.board.history) drawStone(r, c, color);
  if (state.lastMove) drawLastMove(state.lastMove[0], state.lastMove[1]);
  if (state.hover) drawHover(state.hover[0], state.hover[1]);
  updateStatus();
}

function updateStatus() {
  const win = state.board.winner();
  if (win) {
    state.gameOver = true;
    $('status').textContent = win === state.humanColor ? '你赢了！' : '电脑赢了。';
  } else if (state.board.isFull()) {
    state.gameOver = true;
    $('status').textContent = '平局';
  } else if (state.busy) {
    $('status').textContent = '电脑思考中…';
  } else {
    const turn = state.board.sideToMove();
    $('status').textContent = turn === state.humanColor ? '轮到你了' : '轮到电脑';
  }
  $('eval').textContent = state.evalText;
  $('undo').disabled = state.busy || state.board.history.length < 2;
}

function setBusy(v) {
  state.busy = v;
  updateStatus();
}

async function newGame() {
  state.board = new Board(N);
  state.lastMove = null;
  state.gameOver = false;
  state.evalText = '';
  state.hover = null;
  state.humanColor = $('human-color').value === 'black' ? BLACK : WHITE;
  render();
  if (state.humanColor === WHITE) await computerMove();
}

async function computerMove() {
  if (!state.engine || state.gameOver) return;
  const color = 3 - state.humanColor;
  setBusy(true);
  try {
    const temp = Number($('temperature').value) || 0;
    const { move, value } = await state.engine.chooseMove(state.board, color, { temperature: temp });
    if (move) {
      state.board.place(move[0], move[1], color);
      state.lastMove = move;
      const [w, l, d] = valueProbs(value);
      state.evalText = `电脑行棋前判断（电脑视角）→ 胜 ${(w * 100).toFixed(1)}% / `
        + `负 ${(l * 100).toFixed(1)}% / 和 ${(d * 100).toFixed(1)}%`;
    } else {
      state.evalText = '电脑无合法点';
    }
  } catch (err) {
    state.evalText = `推理失败: ${err.message}`;
  } finally {
    setBusy(false);
    render();
  }
}

function toCell(ev) {
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  const c = Math.round((x - MARGIN) / CELL);
  const r = Math.round((y - MARGIN) / CELL);
  if (r < 0 || r >= N || c < 0 || c >= N) return null;
  if (Math.hypot(x - px(c), y - px(r)) > CELL * 0.5) return null;
  return [r, c];
}

async function onClick(ev) {
  if (state.busy || state.gameOver || state.pending) return;
  if (state.board.sideToMove() !== state.humanColor) return;
  const cell = toCell(ev);
  if (!cell || !state.board.isEmpty(cell[0], cell[1])) return;

  state.board.place(cell[0], cell[1], state.humanColor);
  state.lastMove = cell;
  state.hover = null;
  render();
  if (state.board.winner() || state.board.isFull()) return;

  state.pending = true;
  await computerMove();
  state.pending = false;
}

function undo() {
  if (state.busy || state.board.history.length < 2) return;
  state.board.undo();
  state.board.undo();
  const last = state.board.history[state.board.history.length - 1];
  state.lastMove = last ? [last.r, last.c] : null;
  state.gameOver = false;
  state.evalText = '';
  render();
}

async function loadModel() {
  const key = $('model').value;
  setBusy(true);
  state.engine = null;
  $('engine-info').textContent = '加载模型中…';
  try {
    let engine;
    try {
      engine = await NNEngine.load(MODELS[key], { providers: ['webgpu'] });
    } catch (e) {
      engine = await NNEngine.load(MODELS[key], { providers: ['wasm'] });
    }
    state.engine = engine;
    $('engine-info').textContent = `${key} · ${engine.providers.join(',')} · `
      + `${engine.float16 ? 'fp16' : 'fp32'}`;
  } catch (err) {
    const hint = /^https?:\/\//.test(MODELS[key])
      ? '（模型在 R2 跨域加载，请确认 bucket 的 CORS 已允许本站来源）'
      : '';
    $('engine-info').textContent = `模型加载失败: ${err.message}${hint}`;
  } finally {
    setBusy(false);
    updateStatus();
  }
}

function init() {
  setupCanvas();
  $('new-game').addEventListener('click', newGame);
  $('undo').addEventListener('click', undo);
  $('human-color').addEventListener('change', newGame);
  $('model').addEventListener('change', async () => {
    await loadModel();
    await newGame();
  });
  canvas.addEventListener('click', onClick);
  canvas.addEventListener('mousemove', (ev) => {
    if (state.busy || state.gameOver) return;
    const cell = toCell(ev);
    const changed = String(cell) !== String(state.hover);
    state.hover = cell;
    if (changed) render();
  });
  canvas.addEventListener('mouseleave', () => {
    state.hover = null;
    render();
  });
  render();
  loadModel();
}

init();
