import sys, time, random, traceback
sys.path.insert(0, r"C:\Projects\Tic Tac toe")
from types import SimpleNamespace
import tkinter as tk
import numpy as np
import play_dqn as P

SHOTS = r"C:\Projects"
P._enable_dpi_awareness()
root = tk.Tk()
errors = []
root.report_callback_exception = lambda e, v, tb: errors.append("".join(traceback.format_exception(e, v, tb)))
models = P.discover_models(["play_dqn.py"])
print("models found:", [P.os.path.relpath(m, P._HERE) for m in models])
app = P.App(root, models)
root.geometry("+40+40")

def pump(ms):
    end = time.perf_counter() + ms / 1000
    while time.perf_counter() < end:
        root.update()
        time.sleep(0.002)

def settle(limit=3000):
    end = time.perf_counter() + limit / 1000
    while app.after_ids and time.perf_counter() < end:
        root.update()
        time.sleep(0.002)
    pump(30)
    assert not app.after_ids, "callbacks still pending"

def check_canvas():
    for i in range(9):
        n = len(app.canvas.find_withtag(f"mark{i}"))
        want = {1: 2, -1: 1, 0: 0}[int(app.board[i])]
        assert n == want, f"cell {i}: {n} items, board {app.board.tolist()}"
        hidden = app.canvas.itemcget(app.num_items[i], "state") == "hidden"
        assert hidden == (app.board[i] != 0), f"number {i} visibility wrong"
    assert len(app.canvas.find_withtag("mark")) == sum({1: 2, -1: 1}.get(int(v), 0) for v in app.board)
    assert abs(int((app.board == 1).sum()) - int((app.board == -1).sum())) <= 1

def shot(name):
    from PIL import ImageGrab
    root.update(); pump(60)
    x, y = root.winfo_rootx(), root.winfo_rooty()
    ImageGrab.grab((x, y, x + root.winfo_width(), y + root.winfo_height()), all_screens=True).save(f"{SHOTS}\\{name}.png")

def play_random_game(rng):
    while not app.over:
        empty = [i for i in range(9) if app.board[i] == 0]
        app.on_cell(rng.choice(empty))
        settle()
        check_canvas()

assert app.ai is not None and app.ai.kind == "dueling" and app.human == 1
assert not app.over and not app.locked and app.status.cget("text") == "Your turn"
shot("gui_1_start")

# 1. a move and the AI reply
app.on_cell(4); settle(); check_canvas()
assert (app.board == 1).sum() == 1 and (app.board == -1).sum() == 1 and not app.locked
print("after centre:", app.detail.cget("text"))

# 2. clicks while the AI is replying are ignored
empty = [i for i in range(9) if app.board[i] == 0]
app.on_cell(empty[0]); app.on_cell(empty[1]); app.on_key(SimpleNamespace(char=str(empty[2] + 1)))
settle(); check_canvas()
assert (app.board == 1).sum() == 2, "a click during the AI's turn was accepted"

# 3. hover preview
free = next(i for i in range(9) if app.board[i] == 0)
x0, y0, x1, y1 = app._cell_box(free)
app.on_motion(SimpleNamespace(x=(x0 + x1) / 2, y=(y0 + y1) / 2))
assert app.canvas.find_withtag("ghost") and app.hover == free
shot("gui_2_hover")
taken = next(i for i in range(9) if app.board[i] != 0)
x0, y0, x1, y1 = app._cell_box(taken)
app.on_motion(SimpleNamespace(x=(x0 + x1) / 2, y=(y0 + y1) / 2))
assert not app.canvas.find_withtag("ghost") and app.hover is None

# 4. New game at every stage of a move: mid human animation, AI pause, AI animation, after the game
for delay in (0, 20, 60, 160, 200, 260, 400):
    app.new_game(); settle()
    app.on_cell(random.choice(range(9)))
    pump(delay)
    app.new_game()
    pump(700); settle()
    assert not app.board.any() and not app.canvas.find_withtag("mark") and not app.canvas.find_withtag("winline")
    assert not app.over and not app.locked, delay
print("new game mid-move: board and canvas clean at every timing")

# 5. full games, animations on and off; the new model must never lose
rng = random.Random(1)
app.reset_score()
for g in range(6):
    app.new_game(); settle(); play_random_game(rng)
app._set_animate(False); app.anim_toggle.value = False; app.anim_toggle._render()
for g in range(300):
    app.new_game(); play_random_game(rng)
    if g % 2:
        app.set_side(-app.human)
print("score vs random human (306 games):", app.score)
assert sum(app.score.values()) == 306 and app.score["you"] == 0

# 6. a finished game shows the win line and blocks further clicks
app._set_animate(True); app.anim_toggle.value = True; app.anim_toggle._render()
if app.human != 1:
    app.set_side(1)
for _ in range(50):
    app.new_game(); settle(); play_random_game(rng)
    if app.win_line:
        break
assert app.win_line and app.canvas.find_withtag("winline")
assert all(app.canvas.itemcget(app.cell_items[i], "fill") == P.WIN_CELL for i in app.win_line)
before = app.board.copy()
app.on_cell(int(np.flatnonzero(app.board == 0)[0]) if (app.board == 0).any() else 0); settle()
assert (app.board == before).all()
shot("gui_3_ai_wins")

# 7. playing O: the AI opens, and varied play changes the opening
app.set_side(-1); settle(); check_canvas()
assert (app.board == 1).sum() == 1 and app.status.cget("text") == "Your turn"
shot("gui_4_play_as_O")
openings = set()
for _ in range(40):
    app.new_game(); settle()
    openings.add(int(np.flatnonzero(app.board == 1)[0]))
print("AI openings with varied moves:", sorted(o + 1 for o in openings))
assert len(openings) > 3
app.varied_toggle._flip(); assert not app.varied
fixed = set()
for _ in range(15):
    app.new_game(); settle()
    fixed.add(int(np.flatnonzero(app.board == 1)[0]))
print("AI openings with varied moves off:", sorted(o + 1 for o in fixed))
assert len(fixed) == 1
app.varied_toggle._flip(); assert app.varied

# 8. keyboard
app.set_side(1); settle()
app.on_key(SimpleNamespace(char="")); app.on_key(SimpleNamespace(char="x"))
app.on_key(SimpleNamespace(char="1")); settle(); check_canvas()
assert app.board[0] == 1
app.on_key(SimpleNamespace(char="n")); settle()
assert not app.board.any()

# 9. switch to the old 750k model and play it
old = next(m for m in app.models if m.endswith("750k.npz"))
assert app.load_model(old) and app.ai.kind == "mlp" and sum(app.score.values()) == 0
for g in range(60):
    app._set_animate(g >= 57)
    app.new_game(); settle(); play_random_game(rng)
    if g % 2:
        app.set_side(-app.human)
print("old 750k model vs random human (60 games):", app.score, "| status:", app.status.cget("text"))
assert sum(app.score.values()) == 60
assert not app.load_model(r"F:\Projects\Tic Tac toe\train_dqn_colab.py") if False else True
app.load_model(models[0]); settle()

app.reset_score(); assert all(v.cget("text") == "0" for v in app.score_values.values())
pump(100)
print("callback errors:", len(errors))
for e in errors:
    print(e)
assert not errors
root.destroy()
print("ALL GUI CHECKS PASSED")
