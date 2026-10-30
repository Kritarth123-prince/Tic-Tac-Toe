"""
Tic-Tac-Toe vs DQN - desktop GUI (tkinter, built into Python - nothing to install).

Pure NumPy inference: the model loads instantly and every AI move takes well under a millisecond.

Usage
  python play_dqn.py                      # finds tictactoe_dqn.npz next to this script
  python play_dqn.py path/to/model.npz    # any model exported by train_dqn_colab.py

Controls
  click a square or press 1-9 (top-left to bottom-right) - N or F2 starts a new game
"""

import os

# One BLAS thread: a 1-board forward pass gains nothing from threads, and waking them adds latency
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import ctypes
import glob
import math
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from types import SimpleNamespace

import numpy as np

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
_GRID = np.arange(9).reshape(3, 3)
SYMMETRIES = [g.flatten() for k in range(4) for g in (np.rot90(_GRID, k), np.fliplr(np.rot90(_GRID, k)))]

# Varied AI moves pick at random among moves that are exactly as good as the model's best one:
#  * mirror/rotation copies of it on the current board, and
#  * moves whose Q-value is within VARIETY of the best. Checked on the trained model against
#    an exact solver: up to 0.10 never allows a weaker move or skips an immediate win.
VARIETY = 0.05
ANIM_MS = 140        # drawing time of one mark
WIN_ANIM_MS = 260    # drawing time of the winning line
AI_PAUSE_MS = 90     # short pause before the AI's mark, so the two moves read in order
FRAME_MS = 10

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
MODEL_CANDIDATES = [
    os.path.join(_HERE, "tictactoe_dqn.npz"),
    os.path.join(_HERE, "tictactoe_dqn", "tictactoe_dqn.npz"),
    "tictactoe_dqn.npz",
    os.path.join("tictactoe_dqn", "tictactoe_dqn.npz"),
    "/tictactoe_dqn/tictactoe_dqn.npz",
    "/tictactoe_dqn.npz",
    "/content/tictactoe_dqn/tictactoe_dqn.npz",
]
MODEL_GLOBS = [os.path.join(_HERE, "*.npz"), os.path.join(_HERE, "tictactoe_dqn", "*.npz")]

FONT = "Segoe UI" if sys.platform == "win32" else "Helvetica"
BG = "#0f172a"
PANEL = "#1e293b"
PANEL_HOVER = "#2b3a52"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
FAINT = "#475569"
X_COLOUR = "#38bdf8"
O_COLOUR = "#fb7185"
WIN_COLOUR = "#fbbf24"
WIN_CELL = "#3a3320"
ACCENT = "#6366f1"
ACCENT_HOVER = "#818cf8"
GOOD = "#4ade80"


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------

class DQNPlayer:
    """NumPy forward pass of the dueling Q-network exported by train_dqn_colab.py."""

    def __init__(self, path):
        data = np.load(path)
        keys = set(data.files)
        self.groups = {}
        for group in ("body", "value", "adv"):
            layers, i = [], 0
            while f"{group}_{i}_w" in keys:
                layers.append((data[f"{group}_{i}_w"].astype(np.float32),
                               data[f"{group}_{i}_b"].astype(np.float32)))
                i += 1
            if not layers:
                raise ValueError(f"not a tictactoe_dqn model (missing '{group}' layers)")
            self.groups[group] = layers
        # fail now rather than mid-game if the shapes are wrong; also warms up NumPy
        if self.q_values(np.zeros(9, dtype=np.int8), 1).shape != (9,):
            raise ValueError("model does not output 9 Q-values")

    @staticmethod
    def _mlp(x, layers, relu_last):
        for i, (w, b) in enumerate(layers):
            x = x @ w + b
            if relu_last or i < len(layers) - 1:
                x = np.maximum(x, 0.0)
        return x

    def q_values(self, board, me):
        """Q-value of every square for player `me` (+1 = X, -1 = O): +1 win, 0 draw, -1 loss."""
        seen = board * me  # the network sees the board from the side to move
        x = np.concatenate([seen == 1, seen == -1, seen == 0]).astype(np.float32)[None]
        h = self._mlp(x, self.groups["body"], True)
        v = self._mlp(h, self.groups["value"], False)
        a = self._mlp(h, self.groups["adv"], False)
        return (v + a - a.mean(axis=1, keepdims=True))[0]

    def choose(self, board, me, rng=None):
        """Move for player `me`. With an rng, a random pick among moves exactly as good as the
        best one (see VARIETY). Returns (move, q-value of move)."""
        q = self.q_values(board, me).astype(np.float64)
        q[board != 0] = -np.inf
        move = int(np.argmax(q))
        if rng is not None:
            candidates = set(equivalent_moves(board, move))
            candidates.update(np.flatnonzero(q >= q[move] - VARIETY).tolist())
            move = int(rng.choice(sorted(candidates)))
        return move, float(q[move])


def equivalent_moves(board, move):
    """Squares that are the same move as `move` up to a symmetry that leaves the board unchanged."""
    return sorted({int(np.flatnonzero(p == move)[0]) for p in SYMMETRIES if (board[p] == board).all()})


def is_model_file(path):
    try:
        keys = set(np.load(path).files)
    except Exception:
        return False
    return {"body_0_w", "value_0_w", "adv_0_w"} <= keys


def discover_models(argv):
    paths = [a for a in argv[1:] if a.lower().endswith(".npz") and os.path.isfile(a)]
    paths += [p for p in MODEL_CANDIDATES if os.path.isfile(p)]
    for pattern in MODEL_GLOBS:
        paths += sorted(glob.glob(pattern))
    found, seen = [], set()
    for p in paths:
        p = os.path.abspath(p)
        key = os.path.normcase(p)
        if key not in seen and is_model_file(p):
            seen.add(key)
            found.append(p)
    return found


def game_result(board):
    """(winner, winning line): winner is 1/-1, 0 for a draw, None while the game is still going."""
    for line in LINES:
        i, j, k = line
        if board[i] != 0 and board[i] == board[j] == board[k]:
            return int(board[i]), line
    return (0, ()) if (board != 0).all() else (None, ())


def describe(q):
    if q > 0.3:
        return "expects to win"
    if q < -0.3:
        return "expects to lose"
    return "expects a draw"


# ----------------------------------------------------------------------------------------
# Small widgets
# ----------------------------------------------------------------------------------------

def blend(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def ease_out(t):
    return 1 - (1 - t) ** 3


def round_rect(canvas, x0, y0, x1, y1, r, **kw):
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return canvas.create_polygon(pts, smooth=True, **kw)


class FlatButton(tk.Label):
    def __init__(self, master, text, command, bg, hover, fg, font, padx=14, pady=7):
        super().__init__(master, text=text, bg=bg, fg=fg, font=font, padx=padx, pady=pady, cursor="hand2")
        self._bg, self._hover, self._command = bg, hover, command
        self.bind("<Enter>", lambda e: self.config(bg=self._hover))
        self.bind("<Leave>", lambda e: self.config(bg=self._bg))
        self.bind("<Button-1>", lambda e: self._command())

    def set_colours(self, bg, hover, fg):
        self._bg, self._hover = bg, hover
        self.config(bg=bg, fg=fg)


class Toggle(FlatButton):
    def __init__(self, master, text, value, on_change, font):
        self.value, self._text, self._on_change = value, text, on_change
        super().__init__(master, "", self._flip, PANEL, PANEL_HOVER, TEXT, font, padx=12)
        self._render()

    def _flip(self):
        self.value = not self.value
        self._render()
        self._on_change(self.value)

    def _render(self):
        self.config(text=("●  " if self.value else "○  ") + self._text,
                    fg=TEXT if self.value else MUTED)


# ----------------------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------------------

class App:
    def __init__(self, root, models):
        self.root = root
        self.scale = root.winfo_fpixels("1i") / 96.0
        self.cell, self.gap = self.px(112), self.px(10)
        self.size = 3 * self.cell + 4 * self.gap
        self.rng = np.random.default_rng()
        self.models = list(models)
        self.ai, self.model_path = None, None
        self.human = 1
        self.board = np.zeros(9, dtype=np.int8)
        self.over = self.locked = True
        self.win_line = ()
        self.game_id = 0
        self.after_ids = set()
        self.hover = None
        self.ai_note = ""
        self.score = {"you": 0, "draw": 0, "ai": 0}
        self.varied = True
        self.animate = True
        self._build()
        if not (self.models and self.load_model(self.models[0])):
            self.new_game()

    def px(self, v):
        return int(round(v * self.scale))

    # ------------------------------------------------------------------ layout
    def _build(self):
        root, px = self.root, self.px
        root.title("Tic-Tac-Toe vs DQN")
        root.configure(bg=BG)
        root.resizable(False, False)
        font = lambda size, weight="normal": (FONT, size, weight)

        outer = tk.Frame(root, bg=BG, padx=px(22), pady=px(20))
        outer.pack()

        # left: the board
        self.canvas = tk.Canvas(outer, width=self.size, height=self.size, bg=BG,
                                highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=0)
        self.cell_items, self.num_items = [], []
        for i in range(9):
            x0, y0, x1, y1 = self._cell_box(i)
            self.cell_items.append(round_rect(self.canvas, x0, y0, x1, y1, px(14), fill=PANEL, outline=""))
            self.num_items.append(self.canvas.create_text(x0 + px(11), y0 + px(8), text=str(i + 1),
                                                          fill=FAINT, font=font(9), anchor="nw"))
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Leave>", lambda e: self._clear_hover())
        self.canvas.bind("<Button-1>", lambda e: self.on_cell(self._cell_at(e.x, e.y)))

        # right: title, score, status and controls
        side = tk.Frame(outer, bg=BG)
        side.grid(row=0, column=1, sticky="ns", padx=(px(24), 0))
        tk.Label(side, text="Tic-Tac-Toe", font=font(22, "bold"), fg=TEXT, bg=BG).pack(anchor="w")
        tk.Label(side, text="against a self-play Deep Q-Network", font=font(10), fg=MUTED,
                 bg=BG).pack(anchor="w")

        cards = tk.Frame(side, bg=BG)
        cards.pack(fill="x", pady=(px(14), px(14)))
        self.score_titles, self.score_values = {}, {}
        for col, key in enumerate(("you", "draw", "ai")):
            card = tk.Frame(cards, bg=PANEL, padx=px(8), pady=px(6))
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else px(8), 0))
            cards.columnconfigure(col, weight=1, uniform="cards")
            self.score_titles[key] = tk.Label(card, font=font(9, "bold"), fg=MUTED, bg=PANEL)
            self.score_titles[key].pack()
            self.score_values[key] = tk.Label(card, text="0", font=font(20, "bold"), fg=TEXT, bg=PANEL)
            self.score_values[key].pack()

        self.status = tk.Label(side, font=font(15, "bold"), fg=TEXT, bg=BG, anchor="w")
        self.status.pack(fill="x")
        self.detail = tk.Label(side, font=font(10), fg=MUTED, bg=BG, anchor="nw", justify="left", height=2)
        self.detail.pack(fill="x")

        row = tk.Frame(side, bg=BG)
        row.pack(fill="x", pady=(px(12), 0))
        tk.Label(row, text="Play as", font=font(10), fg=MUTED, bg=BG).pack(side="left", padx=(0, px(8)))
        self.side_buttons = {
            1: FlatButton(row, "X  first", lambda: self.set_side(1), PANEL, PANEL_HOVER, TEXT, font(10, "bold")),
            -1: FlatButton(row, "O  second", lambda: self.set_side(-1), PANEL, PANEL_HOVER, TEXT, font(10, "bold")),
        }
        self.side_buttons[1].pack(side="left")
        self.side_buttons[-1].pack(side="left", padx=(px(4), 0))

        row = tk.Frame(side, bg=BG)
        row.pack(fill="x", pady=(px(8), 0))
        FlatButton(row, "New game", self.new_game, ACCENT, ACCENT_HOVER, "#ffffff",
                   font(10, "bold")).pack(side="left")
        FlatButton(row, "Reset score", self.reset_score, PANEL, PANEL_HOVER, TEXT,
                   font(10)).pack(side="left", padx=(px(4), 0))

        row = tk.Frame(side, bg=BG)
        row.pack(fill="x", pady=(px(8), 0))
        self.varied_toggle = Toggle(row, "Varied AI moves", self.varied, self._set_varied, font(10))
        self.varied_toggle.pack(side="left")
        self.anim_toggle = Toggle(row, "Animations", self.animate, self._set_animate, font(10))
        self.anim_toggle.pack(side="left", padx=(px(4), 0))

        row = tk.Frame(side, bg=BG)
        row.pack(fill="x", pady=(px(8), 0))
        tk.Label(row, text="Model", font=font(10), fg=MUTED, bg=BG).pack(side="left", padx=(0, px(8)))
        self.model_button = FlatButton(row, "no model  ▾", self._model_menu, PANEL, PANEL_HOVER,
                                       TEXT, font(10), padx=12)
        self.model_button.pack(side="left", fill="x", expand=True)

        tk.Label(side, text="Keys: 1–9 to move  ·  N or F2 for a new game", font=font(9),
                 fg=FAINT, bg=BG).pack(anchor="w", pady=(px(12), 0))

        # freeze the panel at its natural size so long messages wrap instead of resizing the window
        root.update_idletasks()
        width, height = side.winfo_reqwidth(), side.winfo_reqheight()
        self.detail.config(wraplength=width)
        side.config(width=width, height=height)
        side.pack_propagate(False)

        root.bind("<Key>", self.on_key)
        root.bind("<F2>", lambda e: self.new_game())
        self._update_side_widgets()
        self._update_scores()

    def _cell_box(self, i):
        r, c = divmod(i, 3)
        x0 = self.gap + c * (self.cell + self.gap)
        y0 = self.gap + r * (self.cell + self.gap)
        return x0, y0, x0 + self.cell, y0 + self.cell

    def _cell_centre(self, i):
        x0, y0, x1, y1 = self._cell_box(i)
        return (x0 + x1) / 2, (y0 + y1) / 2

    def _cell_at(self, x, y):
        for i in range(9):
            x0, y0, x1, y1 = self._cell_box(i)
            if x0 <= x < x1 and y0 <= y < y1:
                return i
        return None

    # ------------------------------------------------------------------ scheduling
    def _later(self, ms, fn):
        """root.after that silently drops the call if a new game has started since."""
        game_id, handle = self.game_id, []

        def run():
            self.after_ids.discard(handle[0])
            if game_id == self.game_id:
                fn()

        handle.append(self.root.after(ms, run))
        self.after_ids.add(handle[0])

    def _cancel_pending(self):
        for after_id in self.after_ids:
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
        self.after_ids.clear()

    # ------------------------------------------------------------------ drawing
    def _draw_mark(self, i, player, t, ghost=False):
        """Draw X/O in cell i; t in [0, 1] is how much of the mark is drawn."""
        x0, y0, x1, y1 = self._cell_box(i)
        p = self.cell * 0.25
        width = max(2, round(self.cell * 0.095))
        colour = X_COLOUR if player == 1 else O_COLOUR
        if ghost:
            colour, tags = blend(colour, PANEL_HOVER, 0.72), ("ghost",)
        else:
            tags = ("mark", f"mark{i}")
        if player == 1:
            for k, (ax, ay, bx, by) in enumerate(((x0 + p, y0 + p, x1 - p, y1 - p),
                                                  (x1 - p, y0 + p, x0 + p, y1 - p))):
                s = min(1.0, max(0.0, t * 2 - k))
                if s > 0:
                    self.canvas.create_line(ax, ay, ax + (bx - ax) * s, ay + (by - ay) * s, fill=colour,
                                            width=width, capstyle=tk.ROUND, tags=tags)
        elif t >= 1:
            self.canvas.create_oval(x0 + p, y0 + p, x1 - p, y1 - p, outline=colour, width=width, tags=tags)
        elif t > 0:
            self.canvas.create_arc(x0 + p, y0 + p, x1 - p, y1 - p, start=90, extent=-359.9 * t,
                                   style=tk.ARC, outline=colour, width=width, tags=tags)

    def _animate_mark(self, i, player, then):
        self.canvas.itemconfig(self.num_items[i], state="hidden")
        if not self.animate:
            self._draw_mark(i, player, 1.0)
            then()
            return
        start = time.perf_counter()

        def frame():
            raw = min(1.0, (time.perf_counter() - start) * 1000 / ANIM_MS)
            self.canvas.delete(f"mark{i}")
            self._draw_mark(i, player, 1.0 if raw >= 1 else ease_out(raw))
            if raw < 1:
                self._later(FRAME_MS, frame)
            else:
                then()

        frame()

    def _animate_win_line(self, line):
        (ax, ay), (bx, by) = self._cell_centre(line[0]), self._cell_centre(line[2])
        length = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / length, (by - ay) / length
        ext = self.cell * 0.32
        sx, sy, ex, ey = ax - ux * ext, ay - uy * ext, bx + ux * ext, by + uy * ext
        width = max(3, round(self.cell * 0.07))

        def draw(t):
            self.canvas.delete("winline")
            self.canvas.create_line(sx, sy, sx + (ex - sx) * t, sy + (ey - sy) * t, fill=WIN_COLOUR,
                                    width=width, capstyle=tk.ROUND, tags="winline")

        if not self.animate:
            draw(1.0)
            return
        start = time.perf_counter()

        def frame():
            raw = min(1.0, (time.perf_counter() - start) * 1000 / WIN_ANIM_MS)
            draw(ease_out(raw))
            if raw < 1:
                self._later(FRAME_MS, frame)

        frame()

    # ------------------------------------------------------------------ hover preview
    def _playable(self, i):
        return (i is not None and self.ai is not None and not self.over and not self.locked
                and self.board[i] == 0)

    def on_motion(self, event):
        i = self._cell_at(event.x, event.y)
        if not self._playable(i):
            i = None
        if i == self.hover:
            return
        self._clear_hover()
        if i is not None:
            self.hover = i
            self.canvas.itemconfig(self.cell_items[i], fill=PANEL_HOVER)
            self._draw_mark(i, self.human, 1.0, ghost=True)
            self.canvas.config(cursor="hand2")

    def _clear_hover(self):
        if self.hover is not None:
            self.canvas.itemconfig(self.cell_items[self.hover],
                                   fill=WIN_CELL if self.hover in self.win_line else PANEL)
        self.hover = None
        self.canvas.delete("ghost")
        self.canvas.config(cursor="")

    def _refresh_hover(self):
        px, py = self.canvas.winfo_pointerxy()
        if self.canvas.winfo_containing(px, py) is self.canvas:
            self.on_motion(SimpleNamespace(x=px - self.canvas.winfo_rootx(), y=py - self.canvas.winfo_rooty()))

    # ------------------------------------------------------------------ game flow
    def new_game(self):
        self.game_id += 1
        self._cancel_pending()
        self._clear_hover()
        self.canvas.delete("mark", "winline", "ghost")
        self.board[:] = 0
        self.win_line = ()
        self.ai_note = ""
        for i in range(9):
            self.canvas.itemconfig(self.cell_items[i], fill=PANEL)
            self.canvas.itemconfig(self.num_items[i], state="normal")
        if self.ai is None:
            self.over = self.locked = True
            self._set_status("No model loaded", "Choose a model below (train_dqn_colab.py makes one)", MUTED)
            return
        self.over = False
        if self.human == 1:
            self.locked = False
            self._set_status("Your turn", "You are X · click a square or press 1–9", TEXT)
            self._refresh_hover()
        else:
            self.locked = True
            self._set_status("AI is thinking…", "You are O · the AI opens", TEXT)
            self._later(AI_PAUSE_MS if self.animate else 0, self._ai_move)

    def on_cell(self, i):
        if not self._playable(i):
            return
        self._clear_hover()
        self.board[i] = self.human
        self.locked = True
        self._animate_mark(i, self.human, self._after_human)

    def _after_human(self):
        if self._finish_if_over():
            return
        self._set_status("AI is thinking…", self.detail.cget("text"), TEXT)
        if self.animate:
            self._later(AI_PAUSE_MS, self._ai_move)
        else:
            self._ai_move()

    def _ai_move(self):
        side = -self.human
        start = time.perf_counter()
        move, q = self.ai.choose(self.board, side, self.rng if self.varied else None)
        ms = (time.perf_counter() - start) * 1000
        self.board[move] = side
        self.ai_note = f"AI played {move + 1} · {describe(q)} · {ms:.2f} ms"
        self._animate_mark(move, side, self._after_ai)

    def _after_ai(self):
        if self._finish_if_over():
            return
        self.locked = False
        self._set_status("Your turn", self.ai_note, TEXT)
        self._refresh_hover()

    def _finish_if_over(self):
        winner, line = game_result(self.board)
        if winner is None:
            return False
        self.over = self.locked = True
        self.win_line = line
        if winner == 0:
            key, title, colour = "draw", "Draw!", TEXT
        elif winner == self.human:
            key, title, colour = "you", "You win!", GOOD
        else:
            key, title, colour = "ai", "AI wins!", O_COLOUR if self.human == 1 else X_COLOUR
        self.score[key] += 1
        self._update_scores()
        detail = "Press New game or N to play again"
        if winner == -self.human and self.ai_note:
            detail = f"{self.ai_note} · {detail}"
        self._set_status(title, detail, colour)
        for i in line:
            self.canvas.itemconfig(self.cell_items[i], fill=WIN_CELL)
        if line:
            self._animate_win_line(line)
        return True

    # ------------------------------------------------------------------ controls
    def on_key(self, event):
        if event.char and event.char in "123456789":
            self.on_cell(int(event.char) - 1)
        elif event.char in ("n", "N"):
            self.new_game()

    def set_side(self, side):
        if side == self.human:
            return
        self.human = side
        self._update_side_widgets()
        self.new_game()

    def _set_varied(self, value):
        self.varied = value

    def _set_animate(self, value):
        self.animate = value

    def reset_score(self):
        self.score = {"you": 0, "draw": 0, "ai": 0}
        self._update_scores()

    def _set_status(self, title, detail, colour):
        self.status.config(text=title, fg=colour)
        self.detail.config(text=detail)

    def _update_scores(self):
        for key, label in self.score_values.items():
            label.config(text=str(self.score[key]))

    def _update_side_widgets(self):
        you, ai = (X_COLOUR, O_COLOUR) if self.human == 1 else (O_COLOUR, X_COLOUR)
        self.score_titles["you"].config(text=f"YOU · {'X' if self.human == 1 else 'O'}")
        self.score_titles["draw"].config(text="DRAWS")
        self.score_titles["ai"].config(text=f"AI · {'O' if self.human == 1 else 'X'}")
        self.score_values["you"].config(fg=you)
        self.score_values["ai"].config(fg=ai)
        for side, button in self.side_buttons.items():
            if side == self.human:
                active = X_COLOUR if side == 1 else O_COLOUR
                button.set_colours(active, active, BG)
            else:
                button.set_colours(PANEL, PANEL_HOVER, TEXT)

    # ------------------------------------------------------------------ models
    def _model_label(self, path):
        try:
            rel = os.path.relpath(path, _HERE)
        except ValueError:  # different drive on Windows
            rel = path
        label = (path if rel.startswith("..") else rel).replace("\\", "/")
        return label if len(label) <= 28 else label[:10] + "…" + label[-17:]

    def _model_menu(self):
        menu = tk.Menu(self.root, tearoff=0, bg=PANEL, fg=TEXT, activebackground=ACCENT,
                       activeforeground="#ffffff", font=(FONT, 10))
        current = os.path.normcase(self.model_path) if self.model_path else None
        for path in self.models:
            mark = "✓  " if os.path.normcase(path) == current else "     "
            menu.add_command(label=mark + self._model_label(path), command=lambda p=path: self.load_model(p))
        if self.models:
            menu.add_separator()
        menu.add_command(label="Open another model…", command=self._browse_model)
        x = self.model_button.winfo_rootx()
        y = self.model_button.winfo_rooty() + self.model_button.winfo_height()
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    def _browse_model(self):
        path = filedialog.askopenfilename(parent=self.root, title="Open model", initialdir=_HERE,
                                          filetypes=[("NumPy model", "*.npz"), ("All files", "*.*")])
        if path:
            self.load_model(os.path.abspath(path))

    def load_model(self, path):
        try:
            ai = DQNPlayer(path)
        except Exception as e:
            messagebox.showerror("Could not load model", f"{path}\n\n{e}", parent=self.root)
            return False
        self.ai, self.model_path = ai, path
        if os.path.normcase(path) not in {os.path.normcase(p) for p in self.models}:
            self.models.append(path)
        self.model_button.config(text=f"{self._model_label(path)}  ▾")
        self.reset_score()
        self.new_game()
        return True


def _enable_dpi_awareness():
    """Sharp text on high-DPI Windows screens (sizes are scaled to match in App)."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main():
    _enable_dpi_awareness()
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"Cannot open a window ({e}). On Google Colab use play_colab.py instead.")
        return
    App(root, discover_models(sys.argv))
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 3
    root.geometry(f"+{max(0, x)}+{max(0, y)}")
    root.mainloop()


if __name__ == "__main__":
    main()
