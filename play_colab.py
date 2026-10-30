"""
Play Tic-Tac-Toe against the trained DQN model in Google Colab - clickable board.

How to use
  * Same session as training: paste this whole file into a new cell and run it.
  * New session: run it the same way; if tictactoe_dqn.npz is not found you are asked to
    upload it (it is the file downloaded at the end of training).
  * Or upload this file and run  %run play_colab.py  in a cell.
    (Not `!python play_colab.py` - a clickable board needs the notebook itself.)

Pure NumPy inference: the model loads instantly and each AI move takes under a millisecond.
"""

import os
import time

import numpy as np

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
_GRID = np.arange(9).reshape(3, 3)
SYMMETRIES = [g.flatten() for k in range(4) for g in (np.rot90(_GRID, k), np.fliplr(np.rot90(_GRID, k)))]
SYMBOLS = {1: "X", -1: "O"}
# ipywidgets keeps a button's old text when its description is set to "", so an empty square
# shows a non-breaking space instead (this is what left old X/O marks after "New game")
EMPTY_LABEL = " "
UPLOAD_PATH = "/content/tictactoe_dqn/tictactoe_dqn.npz"
MODEL_CANDIDATES = [
    UPLOAD_PATH,
    "/content/tictactoe_dqn.npz",
    "tictactoe_dqn.npz",
    os.path.join("tictactoe_dqn", "tictactoe_dqn.npz"),
]

# Varied AI moves pick at random among moves that are exactly as good as the model's best one:
#  * mirror/rotation copies of it on the current board, and
#  * moves whose Q-value is within VARIETY of the best. Checked on the trained model against
#    an exact solver: up to 0.10 never allows a weaker move or skips an immediate win.
VARIETY = 0.05

CSS = """<style>
.ttt-cell { font-size: 38px !important; font-weight: 700 !important; border-radius: 10px !important; }
.ttt-x { color: #1d4ed8 !important; }
.ttt-o { color: #dc2626 !important; }
</style>"""
EMPTY_COLOUR, LAST_MOVE_COLOUR, WIN_COLOUR = "#eef2f7", "#dbeafe", "#fde68a"


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
                raise ValueError(f"{path} is not a tictactoe_dqn.npz model (missing '{group}' layers)")
            self.groups[group] = layers

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


def game_result(board):
    """(winner, winning line): winner is 1/-1, 0 for a draw, None while the game is still going."""
    for line in LINES:
        i, j, k = line
        if board[i] != 0 and board[i] == board[j] == board[k]:
            return int(board[i]), line
    return (0, ()) if (board != 0).all() else (None, ())


def describe(q):
    if q > 0.3:
        return "AI expects to win"
    if q < -0.3:
        return "AI expects to lose"
    return "AI expects a draw"


class Game:
    """Game state and rules, independent of the user interface."""

    def __init__(self, ai):
        self.ai = ai
        self.rng = np.random.default_rng()
        self.varied = True
        self.board = np.zeros(9, dtype=np.int8)
        self.human = 1
        self.over = False
        self.last_ai_move = None
        self.win_line = ()
        self.score = {"you": 0, "ai": 0, "draw": 0}

    def new_game(self, human):
        self.board[:] = 0
        self.human, self.over, self.last_ai_move, self.win_line = human, False, None, ()
        if human == -1:
            return self._ai_turn()
        return "Your turn - click a square."

    def play(self, i):
        """The human clicks square i (0-8). Returns the status message to show."""
        if self.over:
            return "Game over - press New game to play again."
        if self.board[i] != 0:
            return "That square is taken - pick an empty one."
        self.board[i] = self.human
        return self._check_end() or self._ai_turn()

    def _ai_turn(self):
        t = time.perf_counter()
        move, q = self.ai.choose(self.board, -self.human, self.rng if self.varied else None)
        ms = (time.perf_counter() - t) * 1000
        self.board[move] = -self.human
        self.last_ai_move = move
        end = self._check_end()
        if end:
            return f"AI played {move + 1}. {end}"
        return f"AI played {move + 1} ({describe(q)}, {ms:.2f} ms). Your turn."

    def _check_end(self):
        winner, self.win_line = game_result(self.board)
        if winner is None:
            return None
        self.over = True
        if winner == 0:
            self.score["draw"] += 1
            return "Draw!"
        if winner == self.human:
            self.score["you"] += 1
            return "You win!"
        self.score["ai"] += 1
        return "AI wins!"


def build_ui(game):
    import ipywidgets as widgets

    out = widgets.Output()
    status = widgets.HTML()
    score = widgets.HTML()
    side = widgets.ToggleButtons(options=[("Play X (you start)", 1), ("Play O (AI starts)", -1)],
                                 value=1, style={"button_width": "150px"})
    new_button = widgets.Button(description="New game", button_style="primary")
    varied = widgets.Checkbox(value=game.varied, description="Varied AI moves", indent=False)
    cells = [widgets.Button(description=EMPTY_LABEL, layout=widgets.Layout(width="86px", height="86px"))
             for _ in range(9)]
    for cell in cells:
        cell.add_class("ttt-cell")
    grid = widgets.GridBox(cells, layout=widgets.Layout(grid_template_columns="repeat(3, 86px)",
                                                        grid_gap="6px", margin="10px 0"))

    def render(message):
        for i, cell in enumerate(cells):
            v = int(game.board[i])
            cell.description = SYMBOLS.get(v, EMPTY_LABEL)
            cell.remove_class("ttt-x")
            cell.remove_class("ttt-o")
            if v:
                cell.add_class("ttt-x" if v == 1 else "ttt-o")
            if i in game.win_line:
                cell.style.button_color = WIN_COLOUR
            elif i == game.last_ai_move:
                cell.style.button_color = LAST_MOVE_COLOUR
            else:
                cell.style.button_color = EMPTY_COLOUR
        status.value = f"<b>{message}</b>"
        s = game.score
        score.value = f"Score - you {s['you']} &nbsp;|&nbsp; AI {s['ai']} &nbsp;|&nbsp; draws {s['draw']}"

    @out.capture()
    def on_cell(i):
        render(game.play(i))

    @out.capture()
    def on_new_game(*_):
        render(game.new_game(side.value))

    @out.capture()
    def on_varied(change):
        game.varied = change["new"]

    for i, cell in enumerate(cells):
        cell.on_click(lambda _button, i=i: on_cell(i))
    new_button.on_click(on_new_game)
    side.observe(on_new_game, names="value")
    varied.observe(on_varied, names="value")

    render(game.new_game(side.value))
    return widgets.VBox([widgets.HTML("<h3>Tic-Tac-Toe vs DQN</h3>"),
                         widgets.HBox([side, new_button]), varied, grid, status, score, out])


def locate_model():
    for path in MODEL_CANDIDATES:
        if os.path.isfile(path):
            return path
    try:
        from google.colab import files
    except ImportError:
        return None
    print("tictactoe_dqn.npz not found - please upload it (the file downloaded at the end of training).")
    for name, data in files.upload().items():
        if name.endswith(".npz"):
            os.makedirs(os.path.dirname(UPLOAD_PATH), exist_ok=True)
            with open(UPLOAD_PATH, "wb") as f:
                f.write(data)
            return UPLOAD_PATH
    return None


def main():
    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except ImportError:
        in_notebook = False
    if not in_notebook:
        print("This runner needs a notebook (Colab or Jupyter). On a desktop use:  python play_dqn.py")
        return

    path = locate_model()
    if path is None:
        print("No model found. Train with train_dqn_colab.py first, or upload tictactoe_dqn.npz.")
        return
    from IPython.display import HTML, display
    display(HTML(CSS))
    display(build_ui(Game(DQNPlayer(path))))
    print(f"Model: {path}")


if __name__ == "__main__":
    main()
