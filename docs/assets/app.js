// Tic-Tac-Toe DQN - web player: board, game flow and settings.
import { loadModel, gameResult } from "./model.js";

const MARK_MS = 240;      // drawing time of one mark
const AI_PAUSE_MS = 150;  // pause before the network answers
const CLEAR_MS = 200;     // old marks fading out on a new game
const WIN_MS = 560;       // winning line
const STORAGE_KEY = "ttt-dqn:v1";
const SVG_NS = "http://www.w3.org/2000/svg";
const MARK_PATHS = {
  1: ["M30 30 L70 70", "M70 30 L30 70"],
  [-1]: ["M50 28 A22 22 0 1 1 50 72 A22 22 0 1 1 50 28"],
};

const root = document.documentElement;
const byId = (id) => document.getElementById(id);
const els = {
  board: byId("board"),
  cells: byId("cells"),
  winline: byId("winline"),
  winPath: byId("winline-path"),
  winGradient: byId("win-gradient"),
  loadbar: byId("loadbar"),
  loadFill: byId("loadbar-fill"),
  statusTitle: byId("status-title"),
  statusDetail: byId("status-detail"),
  outlookValue: byId("outlook-value"),
  meter: byId("meter"),
  meterDot: byId("meter-dot"),
  readout: byId("readout"),
  side: byId("side"),
  varied: byId("varied"),
  animate: byId("animate"),
  theme: byId("theme-toggle"),
  meta: byId("model-meta"),
  scores: { you: byId("score-you"), draw: byId("score-draw"), ai: byId("score-ai") },
  labels: { you: byId("label-you"), ai: byId("label-ai") },
};

const state = {
  model: null,
  board: new Int8Array(9),
  human: 1,
  over: true,
  locked: true,
  gen: 0,              // bumped by every new game; stale timers check it and stop
  timers: new Set(),
  score: { you: 0, draw: 0, ai: 0 },
  varied: true,
  animate: !matchMedia("(prefers-reduced-motion: reduce)").matches,
  theme: root.dataset.theme === "light" ? "light" : "dark",
};
const cells = [];
let winLineToken = 0;

// ------------------------------------------------------------------ settings

function loadSettings() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch { saved = null; }
  if (!saved || typeof saved !== "object") return;
  if (saved.human === 1 || saved.human === -1) state.human = saved.human;
  if (typeof saved.varied === "boolean") state.varied = saved.varied;
  if (typeof saved.animate === "boolean") state.animate = saved.animate;
  if (saved.score && typeof saved.score === "object") {
    for (const key of ["you", "draw", "ai"]) {
      const value = saved.score[key];
      if (Number.isInteger(value) && value >= 0) state.score[key] = value;
    }
  }
}

function saveSettings() {
  const { human, varied, animate, theme, score } = state;
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ human, varied, animate, theme, score })); } catch { /* private mode */ }
}

// ------------------------------------------------------------------ scheduling

/** setTimeout that is dropped when a new game starts; runs at once for 0 ms. */
function later(ms, fn) {
  if (ms <= 0) { fn(); return; }
  const gen = state.gen;
  const id = setTimeout(() => {
    state.timers.delete(id);
    if (gen === state.gen) fn();
  }, ms);
  state.timers.add(id);
}

function cancelTimers() {
  for (const id of state.timers) clearTimeout(id);
  state.timers.clear();
}

const motion = (ms) => (state.animate ? ms : 0);

// ------------------------------------------------------------------ rendering

function svgMark(player, kind) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 100 100");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  svg.setAttribute("class", `${kind} ${kind}-${player === 1 ? "x" : "o"}`);
  for (const d of MARK_PATHS[player]) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    path.setAttribute("pathLength", "1");
    svg.append(path);
  }
  return svg;
}

function buildCells() {
  for (let i = 0; i < 9; i++) {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "cell";
    const num = document.createElement("span");
    num.className = "cell-num";
    num.setAttribute("aria-hidden", "true");
    num.textContent = String(i + 1);
    cell.append(svgMark(1, "ghost"), svgMark(-1, "ghost"), num);
    cell.addEventListener("click", () => play(i));
    els.cells.append(cell);
    cells.push(cell);
  }
}

function drawMark(i, player) {
  const mark = svgMark(player, "mark");
  if (state.animate) mark.classList.add("is-drawing");
  cells[i].append(mark);
  cells[i].classList.add("is-filled");
}

function updateCellLabels() {
  cells.forEach((cell, i) => {
    const v = state.board[i];
    cell.setAttribute("aria-label", `Square ${i + 1}, ${v === 1 ? "X" : v === -1 ? "O" : "empty"}`);
  });
}

function syncBoard() {
  const playable = Boolean(state.model) && !state.over && !state.locked;
  els.board.classList.toggle("is-playable", playable);
  els.board.dataset.human = state.human === 1 ? "x" : "o";
  let xs = 0;
  let os = 0;
  for (const v of state.board) { if (v === 1) xs++; else if (v === -1) os++; }
  root.dataset.turn = !state.model || state.over ? "none" : xs === os ? "x" : "o";
}

function setStatus(title, detail = "", { tone = "", dots = false } = {}) {
  const changed = els.statusTitle.textContent !== title;
  els.statusTitle.textContent = title;
  els.statusTitle.dataset.tone = tone;
  els.statusDetail.textContent = detail;
  if (dots) {
    const span = document.createElement("span");
    span.className = "dots";
    span.setAttribute("aria-hidden", "true");
    span.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
    els.statusDetail.append(span);
  }
  if (changed && state.animate && els.statusTitle.animate) {
    els.statusTitle.animate(
      [{ opacity: 0, transform: "translateY(5px)" }, { opacity: 1, transform: "none" }],
      { duration: 280, easing: "cubic-bezier(.22, 1, .36, 1)" },
    );
  }
}

function describe(q) {
  if (q > 0.3) return "Expects to win";
  if (q < -0.3) return "Expects to lose";
  return "Expects a draw";
}

function formatMs(ms) {
  return ms < 0.1 ? "<0.1 ms" : `${ms.toFixed(1)} ms`;
}

function setOutlook(q = null, move = null, ms = null) {
  if (q === null) {
    els.meter.classList.add("is-empty");
    els.meterDot.style.left = "50%";
    els.meter.setAttribute("aria-valuenow", "0");
    els.meter.setAttribute("aria-valuetext", "No move yet");
    els.outlookValue.textContent = "No move yet";
    els.readout.textContent = "Q-value, chosen square and move time appear here";
    return;
  }
  const clamped = Math.max(-1, Math.min(1, q));
  const shown = Math.abs(q) < 0.0005 ? 0 : q;
  els.meter.classList.remove("is-empty");
  els.meterDot.style.left = `${((clamped + 1) / 2) * 100}%`;
  els.meter.setAttribute("aria-valuenow", clamped.toFixed(2));
  els.meter.setAttribute("aria-valuetext", describe(q));
  els.outlookValue.textContent = describe(q);
  els.readout.textContent = `Q ${shown < 0 ? "−" : "+"}${Math.abs(shown).toFixed(3)} · square ${move + 1} · ${formatMs(ms)}`;
}

function renderScores(bumpKey = null) {
  for (const key of ["you", "draw", "ai"]) els.scores[key].textContent = String(state.score[key]);
  if (bumpKey && state.animate) {
    const el = els.scores[bumpKey];
    el.classList.remove("is-bump");
    void el.offsetWidth; // restart the animation
    el.classList.add("is-bump");
  }
}

function glyph(player) {
  const span = document.createElement("span");
  span.className = `glyph glyph-${player === 1 ? "x" : "o"}`;
  span.textContent = player === 1 ? "X" : "O";
  return span;
}

function renderSide() {
  els.labels.you.replaceChildren("You · ", glyph(state.human));
  els.labels.ai.replaceChildren("Network · ", glyph(-state.human));
  els.side.dataset.value = state.human === 1 ? "x" : "o";
  for (const button of els.side.querySelectorAll("button[data-side]")) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.side) === state.human));
  }
  root.dataset.ai = state.human === 1 ? "o" : "x";
}

function renderSwitches() {
  els.varied.setAttribute("aria-checked", String(state.varied));
  els.animate.setAttribute("aria-checked", String(state.animate));
}

function applyMotion() {
  root.classList.toggle("no-motion", !state.animate);
  if (!state.animate) for (const mark of els.cells.querySelectorAll(".mark.is-leaving")) mark.remove();
}

function applyTheme() {
  root.dataset.theme = state.theme;
  els.theme.setAttribute("aria-label", `Switch to ${state.theme === "dark" ? "light" : "dark"} theme`);
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", state.theme === "dark" ? "#0b0c0f" : "#f3efe7");
}

function showWinLine(line) {
  const centre = (i) => [50 + (i % 3) * 100, 50 + Math.floor(i / 3) * 100];
  const [ax, ay] = centre(line[0]);
  const [bx, by] = centre(line[2]);
  const length = Math.hypot(bx - ax, by - ay);
  const ux = (bx - ax) / length;
  const uy = (by - ay) / length;
  const ext = 34;
  const [x1, y1, x2, y2] = [ax - ux * ext, ay - uy * ext, bx + ux * ext, by + uy * ext].map((v) => v.toFixed(1));
  els.winPath.setAttribute("d", `M${x1} ${y1} L${x2} ${y2}`);
  for (const [name, value] of Object.entries({ x1, y1, x2, y2 })) els.winGradient.setAttribute(name, value);
  winLineToken += 1;
  els.winline.classList.remove("is-leaving", "is-visible");
  void els.winline.getBoundingClientRect(); // start the stroke from zero
  els.winline.classList.add("is-visible");
}

function hideWinLine() {
  const svg = els.winline;
  if (!svg.classList.contains("is-visible")) return;
  svg.classList.remove("is-visible");
  const token = ++winLineToken;
  const reset = () => {
    if (token !== winLineToken) return;
    svg.classList.remove("is-leaving");
    els.winPath.removeAttribute("d");
  };
  if (state.animate) {
    svg.classList.add("is-leaving");
    setTimeout(reset, CLEAR_MS);
  } else {
    reset();
  }
}

function clearBoard() {
  els.board.classList.remove("is-over", "is-draw");
  for (const cell of cells) {
    cell.classList.remove("is-filled", "is-last", "is-win");
    for (const mark of cell.querySelectorAll(".mark")) {
      if (!state.animate) {
        mark.remove();
      } else if (!mark.classList.contains("is-leaving")) {
        mark.classList.remove("is-drawing");
        mark.classList.add("is-leaving");
        setTimeout(() => mark.remove(), CLEAR_MS);
      }
    }
  }
  hideWinLine();
}

// ------------------------------------------------------------------ game flow

function newGame() {
  state.gen += 1;
  cancelTimers();
  const fading = state.animate && els.cells.querySelector(".mark") !== null;
  state.board.fill(0);
  state.over = false;
  state.locked = true;
  clearBoard();
  setOutlook();
  renderScores();
  updateCellLabels();
  syncBoard();
  if (!state.model) return;

  const settle = fading ? CLEAR_MS : 0;
  if (state.human === 1) {
    setStatus("Your turn", "You play X. Pick any square.");
    later(settle, () => { state.locked = false; syncBoard(); });
  } else {
    setStatus("Thinking", "The network opens as X", { dots: true });
    later(settle + motion(AI_PAUSE_MS), aiMove);
  }
}

function play(i) {
  if (!state.model || state.over || state.locked || state.board[i] !== 0) return;
  state.board[i] = state.human;
  state.locked = true;
  drawMark(i, state.human);
  updateCellLabels();
  syncBoard();
  if (concludeIfOver()) return;
  setStatus("Thinking", "", { dots: true });
  later(motion(Math.round(MARK_MS * 0.6) + AI_PAUSE_MS), aiMove);
}

function aiMove() {
  const me = -state.human;
  let choice;
  const start = performance.now();
  try {
    choice = state.model.choose(state.board, me, state.varied ? Math.random : null);
  } catch (error) {
    console.error(error);
    setStatus("Something went wrong", String(error.message || error), { tone: "error" });
    return;
  }
  const ms = performance.now() - start;
  const { move, q } = choice;
  state.board[move] = me;
  for (const cell of cells) cell.classList.remove("is-last");
  drawMark(move, me);
  cells[move].classList.add("is-last");
  updateCellLabels();
  setOutlook(q, move, ms);
  if (concludeIfOver()) return;
  state.locked = false;
  syncBoard();
  setStatus("Your turn", `The network played square ${move + 1}.`);
}

function concludeIfOver() {
  const { winner, line } = gameResult(state.board);
  if (winner === null) return false;
  state.over = true;
  state.locked = true;
  const key = winner === 0 ? "draw" : winner === state.human ? "you" : "ai";
  state.score[key] += 1;
  saveSettings();
  syncBoard();
  const titles = { you: "You win", draw: "Draw", ai: "The network wins" };
  const details = {
    you: "You beat the network.",
    draw: "Neither side could break through.",
    ai: "It found the winning line.",
  };
  const hint = matchMedia("(hover: hover) and (pointer: fine)").matches
    ? "Press N to play again."
    : "Tap New game to play again.";
  setStatus(titles[key], `${details[key]} ${hint}`, { tone: key });
  // the line, dimming and score land once the final mark has been drawn
  later(motion(MARK_MS), () => showResult(key, line));
  return true;
}

function showResult(key, line) {
  els.board.classList.add(key === "draw" ? "is-draw" : "is-over");
  if (line) {
    for (const i of line) cells[i].classList.add("is-win");
    showWinLine(line);
  }
  renderScores(key);
}

// ------------------------------------------------------------------ controls

function resetScore() {
  state.score = { you: 0, draw: 0, ai: 0 };
  saveSettings();
  renderScores();
}

function bindControls() {
  // the same actions appear in more than one place (desktop panel, phone score bar, board)
  for (const button of document.querySelectorAll('[data-action="new-game"]')) button.addEventListener("click", newGame);
  for (const button of document.querySelectorAll('[data-action="reset-score"]')) button.addEventListener("click", resetScore);

  els.side.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-side]");
    if (!button) return;
    const side = Number(button.dataset.side);
    if (side === state.human) return;
    state.human = side;
    saveSettings();
    renderSide();
    newGame();
  });

  els.varied.addEventListener("click", () => {
    state.varied = !state.varied;
    renderSwitches();
    saveSettings();
  });

  els.animate.addEventListener("click", () => {
    state.animate = !state.animate;
    renderSwitches();
    applyMotion();
    saveSettings();
  });

  els.theme.addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    applyTheme();
    saveSettings();
  });

  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
    if (event.key.length === 1 && event.key >= "1" && event.key <= "9") play(Number(event.key) - 1);
    else if (event.key === "n" || event.key === "N") newGame();
  });
}

// ------------------------------------------------------------------ start

function describeModel(manifest) {
  const { positions, optimal } = manifest.evaluation || {};
  const parts = [`${manifest.parameters.toLocaleString("en-GB")} parameters`];
  if (positions) parts.push(`optimal in ${optimal.toLocaleString("en-GB")} of ${positions.toLocaleString("en-GB")} positions`);
  parts.push(`${(manifest.bytes / 1e6).toFixed(1)} MB ${manifest.dtype} weights`);
  return parts.join(" · ");
}

async function start() {
  root.style.setProperty("--mark-ms", `${MARK_MS}ms`);
  root.style.setProperty("--clear-ms", `${CLEAR_MS}ms`);
  root.style.setProperty("--win-ms", `${WIN_MS}ms`);
  loadSettings();
  buildCells();
  bindControls();
  renderSide();
  renderSwitches();
  renderScores();
  applyMotion();
  applyTheme();
  updateCellLabels();
  setOutlook();
  syncBoard();
  setStatus("Loading the network", "Fetching weights");

  try {
    const { model, manifest } = await loadModel(new URL("../model/", import.meta.url), (received, total) => {
      els.loadFill.style.transform = `scaleX(${Math.min(1, received / total)})`;
      els.statusDetail.textContent = `${(received / 1e6).toFixed(1)} of ${(total / 1e6).toFixed(1)} MB`;
    });
    // let the JavaScript engine optimise the forward pass before the first real move
    const sample = new Int8Array([0, 0, 0, 0, 1, 0, 0, 0, 0]);
    for (let k = 0; k < 30; k++) model.choose(sample, -1);
    state.model = model;
    els.loadFill.style.transform = "scaleX(1)";
    els.loadbar.classList.add("is-done");
    els.meta.textContent = describeModel(manifest);
    newGame();
  } catch (error) {
    console.error(error);
    els.loadbar.classList.add("is-error");
    const detail = location.protocol === "file:"
      ? "Browsers block this when the page is opened as a file. Serve the folder instead: python -m http.server -d docs"
      : String(error.message || error);
    setStatus("The model couldn't load", detail, { tone: "error" });
  }
}

start();
