// Tic-Tac-Toe DQN - browser inference, no dependencies.
// Mirrors DQNPlayer in play_dqn.py: same input encoding, dueling head and varied-move rule.

// Varied moves pick at random among moves exactly as good as the best one: mirror/rotation
// copies of it on the current board, and moves whose Q-value is within VARIETY of the best.
// Checked against an exact solver by tools/export_web_model.py.
export const VARIETY = 0.05;

export const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];

// The 8 symmetries of the grid: newBoard[i] = board[SYMMETRIES[k][i]]
export const SYMMETRIES = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 0, 5, 4, 3, 8, 7, 6],
  [2, 5, 8, 1, 4, 7, 0, 3, 6], [8, 5, 2, 7, 4, 1, 6, 3, 0],
  [8, 7, 6, 5, 4, 3, 2, 1, 0], [6, 7, 8, 3, 4, 5, 0, 1, 2],
  [6, 3, 0, 7, 4, 1, 8, 5, 2], [0, 3, 6, 1, 4, 7, 2, 5, 8],
];

/** { winner: 1 | -1 | 0 (draw) | null (still playing), line: [a, b, c] | null } */
export function gameResult(board) {
  for (const line of LINES) {
    const [a, b, c] = line;
    if (board[a] !== 0 && board[a] === board[b] && board[a] === board[c]) {
      return { winner: board[a], line };
    }
  }
  for (let i = 0; i < 9; i++) if (board[i] === 0) return { winner: null, line: null };
  return { winner: 0, line: null };
}

/** Squares that are the same move as `move` up to a symmetry that leaves the board unchanged. */
export function equivalentMoves(board, move) {
  const moves = new Set();
  for (const p of SYMMETRIES) {
    let unchanged = true;
    for (let i = 0; i < 9 && unchanged; i++) unchanged = board[p[i]] === board[i];
    if (unchanged) moves.add(p.indexOf(move));
  }
  return [...moves].sort((a, b) => a - b);
}

function dense(x, layer, relu) {
  const { w, b, inDim, outDim } = layer;
  const out = new Float64Array(outDim);
  out.set(b);
  for (let i = 0; i < inDim; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = i * outDim;
    for (let j = 0; j < outDim; j++) out[j] += xi * w[row + j];
  }
  if (relu) for (let j = 0; j < outDim; j++) if (out[j] < 0) out[j] = 0;
  return out;
}

function mlp(x, layers, reluLast) {
  let h = x;
  layers.forEach((layer, i) => { h = dense(h, layer, reluLast || i < layers.length - 1); });
  return h;
}

export class DQNModel {
  constructor(groups) {
    this.groups = groups;
  }

  /** Q-value of every square for player `me` (+1 = X, -1 = O): +1 win, 0 draw, -1 loss. */
  qValues(board, me) {
    const x = new Float64Array(27);
    for (let i = 0; i < 9; i++) {
      const seen = board[i] * me; // the network sees the board from the side to move
      x[i] = seen === 1 ? 1 : 0;
      x[9 + i] = seen === -1 ? 1 : 0;
      x[18 + i] = seen === 0 ? 1 : 0;
    }
    const h = mlp(x, this.groups.body, true);
    const v = mlp(h, this.groups.value, false)[0];
    const a = mlp(h, this.groups.adv, false);
    let mean = 0;
    for (let j = 0; j < 9; j++) mean += a[j];
    mean /= 9;
    const q = new Float64Array(9);
    for (let j = 0; j < 9; j++) q[j] = v + a[j] - mean;
    return q;
  }

  /** Move for player `me`. Pass `random` (a () => [0, 1) function) for varied moves. */
  choose(board, me, random = null) {
    const q = this.qValues(board, me);
    let best = -1;
    for (let j = 0; j < 9; j++) {
      if (board[j] !== 0) continue;
      if (best < 0 || q[j] > q[best]) best = j;
    }
    if (best < 0) throw new Error("no legal move");
    let move = best;
    if (random) {
      const candidates = new Set(equivalentMoves(board, best));
      for (let j = 0; j < 9; j++) if (board[j] === 0 && q[j] >= q[best] - VARIETY) candidates.add(j);
      const list = [...candidates].sort((a, b) => a - b);
      move = list[Math.min(list.length - 1, Math.floor(random() * list.length))];
    }
    return { move, q: q[move], best, values: q };
  }
}

let halfTable = null;

function halfToFloatTable() {
  if (halfTable) return halfTable;
  halfTable = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x03ff;
    if (exp === 0) halfTable[h] = sign * frac * 2 ** -24;
    else if (exp === 31) halfTable[h] = frac ? NaN : sign * Infinity;
    else halfTable[h] = sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  return halfTable;
}

/** Build a model from model.json and the raw bytes of weights.bin. */
export function parseModel(manifest, buffer) {
  if (manifest.format !== "tictactoe-dqn-web" || manifest.version !== 1) {
    throw new Error("unsupported model.json format");
  }
  if (buffer.byteLength !== manifest.bytes) {
    throw new Error(`weights.bin is ${buffer.byteLength} bytes, expected ${manifest.bytes}`);
  }
  const n = manifest.parameters;
  const view = new DataView(buffer);
  const params = new Float32Array(n);
  if (manifest.dtype === "float16") {
    if (n * 2 !== buffer.byteLength) throw new Error("weights.bin size does not match float16");
    const table = halfToFloatTable();
    for (let i = 0; i < n; i++) params[i] = table[view.getUint16(i * 2, true)];
  } else if (manifest.dtype === "float32") {
    if (n * 4 !== buffer.byteLength) throw new Error("weights.bin size does not match float32");
    for (let i = 0; i < n; i++) params[i] = view.getFloat32(i * 4, true);
  } else {
    throw new Error(`unsupported dtype ${manifest.dtype}`);
  }

  const groups = {};
  for (const name of ["body", "value", "adv"]) {
    const specs = manifest.layers[name];
    if (!Array.isArray(specs) || specs.length === 0) throw new Error(`model.json has no '${name}' layers`);
    groups[name] = specs.map((s) => {
      if (s.w + s.in * s.out > n || s.b + s.out > n) throw new Error(`layer of '${name}' is out of range`);
      return { inDim: s.in, outDim: s.out, w: params.subarray(s.w, s.w + s.in * s.out), b: params.subarray(s.b, s.b + s.out) };
    });
  }
  const chain = (layers, first) => layers.every((l, i) => l.inDim === (i === 0 ? first : layers[i - 1].outDim));
  const bodyOut = groups.body[groups.body.length - 1].outDim;
  if (!chain(groups.body, 27) || !chain(groups.value, bodyOut) || !chain(groups.adv, bodyOut)
      || groups.value[groups.value.length - 1].outDim !== 1 || groups.adv[groups.adv.length - 1].outDim !== 9) {
    throw new Error("model.json layer shapes do not fit together");
  }
  for (let i = 0; i < n; i++) if (!Number.isFinite(params[i])) throw new Error("weights contain invalid numbers");
  return new DQNModel(groups);
}

async function readWithProgress(response, total, onProgress) {
  if (!response.body || !response.body.getReader) {
    const buffer = await response.arrayBuffer();
    onProgress?.(buffer.byteLength, total);
    return buffer;
  }
  // count decompressed bytes against the manifest size, so gzip transfer never skews progress
  const reader = response.body.getReader();
  const bytes = new Uint8Array(total);
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length > total) throw new Error("weights.bin is larger than model.json says");
    bytes.set(value, received);
    received += value.length;
    onProgress?.(received, total);
  }
  if (received !== total) throw new Error(`weights.bin download stopped at ${received} of ${total} bytes`);
  return bytes.buffer;
}

async function checkSha256(buffer, expected) {
  if (!expected || !globalThis.crypto?.subtle) return; // needs https or localhost
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  if (hex !== expected) throw new Error("weights.bin does not match model.json (checksum mismatch)");
}

/** Download model.json + weights.bin from `baseUrl`. onProgress(receivedBytes, totalBytes). */
export async function loadModel(baseUrl, onProgress) {
  const manifestResponse = await fetch(new URL("model.json", baseUrl), { cache: "no-cache" });
  if (!manifestResponse.ok) throw new Error(`model.json: HTTP ${manifestResponse.status}`);
  const manifest = await manifestResponse.json();

  const weightsUrl = new URL(manifest.weights, baseUrl);
  weightsUrl.searchParams.set("v", String(manifest.sha256 || "").slice(0, 12)); // new weights, new URL
  const response = await fetch(weightsUrl);
  if (!response.ok) throw new Error(`weights.bin: HTTP ${response.status}`);
  const buffer = await readWithProgress(response, manifest.bytes, onProgress);
  await checkSha256(buffer, manifest.sha256);
  return { model: parseModel(manifest, buffer), manifest };
}
