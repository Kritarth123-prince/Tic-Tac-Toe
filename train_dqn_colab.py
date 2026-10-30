"""
Tic-Tac-Toe - self-play Deep Q-Network (DQN) training, optimised for Google Colab (T4 GPU)
==========================================================================================

How to run on Colab
  1. Runtime -> Change runtime type -> T4 GPU
  2. Paste this whole file into one cell and run it
     (or upload it and run:  !python train_dqn_colab.py)
  3. Outputs land in /content/tictactoe_dqn/
       tictactoe_dqn.npz   <- weights used by play_dqn.py (NumPy only, instant moves)
       tictactoe_dqn.pt    <- full PyTorch checkpoint
       training_curves.png, history.json

Method
  * Negamax self-play DQN: the board is always seen from the side to move
    (+1 = me, -1 = opponent), so ONE network learns to play both X and O.
        y = r                                                   if the move ended the game
        y = -gamma * Q_target(s', argmax_a' Q_online(s', a'))   otherwise
    (s' belongs to the opponent, so its value is negated).
  * Double DQN, dueling head, Polyak-averaged target network, illegal-move masking.
  * Thousands of games simulated in parallel directly on the GPU (no CPU bottleneck).
  * GPU replay buffer with random 8-fold board symmetry augmentation.
  * FP16 mixed precision (T4 tensor cores) with a fused Adam optimiser.
  * Measured against an exact minimax solver: "accuracy" is the share of ALL 4,520
    reachable positions where the network picks a game-theoretically optimal move.
    The checkpoint with the best accuracy is kept.
"""

import contextlib
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_COLAB = os.path.isdir("/content")

CONFIG = {
    "seed": 42,
    "out_dir": "/content/tictactoe_dqn" if IN_COLAB else "./tictactoe_dqn",
    "allow_cpu": False,             # training is meant for the T4; set True to force a CPU run
    # network
    "hidden": 512,
    "head_hidden": 256,
    # self-play and replay
    "n_envs": 4096,                 # games played in parallel on the GPU
    "buffer_size": 2_000_000,
    "warmup_transitions": 200_000,
    "iterations": 40_000,           # one iteration = one move in every parallel game
    "updates_per_iter": 2,
    "batch_size": 8192,
    # reinforcement learning
    "gamma": 0.9,
    "tau": 0.005,                   # Polyak factor for the target network
    "eps_start": 1.0,
    "eps_decay_frac": 0.4,          # share of training over which epsilon decays to its floor
    "eps_floor_min": 0.02,          # every game keeps its own exploration floor, spread
    "eps_floor_max": 0.5,           # evenly between these two values (state coverage)
    # optimisation
    "lr": 1e-3,
    "lr_min": 1e-5,
    "grad_clip": 10.0,
    "use_amp": True,
    # evaluation
    "eval_every": 250,
    "eval_games": 4096,             # games per match during training
    "final_eval_games": 50_000,     # games per match for the final report
    "early_stop_evals": 12,         # stop once perfect for this many evaluations in a row...
    "early_stop_min_frac": 0.5,     # ...but never before this share of training is done
    # outputs
    "save_to_drive": False,         # True: copy outputs to MyDrive/tictactoe_dqn (asks for Drive access)
    "download_at_end": True,        # Colab notebook only: download tictactoe_dqn.npz
}

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_ctx(amp):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if amp else contextlib.nullcontext()


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def clock():
    return time.strftime("%H:%M:%S")


def fmt_duration(sec):
    sec = int(sec)
    return f"{sec // 3600:d}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


def encode(boards):
    """Canonical boards [N, 9] (+1 me, -1 opponent, 0 empty) -> one-hot features [N, 27]."""
    return torch.cat([boards == 1, boards == -1, boards == 0], dim=1).float()


def is_win(boards, lines):
    """True where the side that just moved (+1) owns a full line."""
    return (boards[:, lines].sum(-1) == 3).any(1)


def board_symmetries():
    """The 8 symmetries of the 3x3 grid. new_board[i] = old_board[perms[k, i]];
    a move at old position a lands on new position inv[k, a]."""
    base = np.arange(9).reshape(3, 3)
    perms = []
    for k in range(4):
        rot = np.rot90(base, k)
        perms.append(rot.flatten())
        perms.append(np.fliplr(rot).flatten())
    perms = np.stack(perms)
    inv = np.argsort(perms, axis=1)
    return perms, inv


# ----------------------------------------------------------------------------------------
# Exact solver (used only for evaluation, never for training)
# ----------------------------------------------------------------------------------------

def _has_line(b, p):
    return any(b[i] == p and b[j] == p and b[k] == p for i, j, k in LINES)


def solve_tictactoe():
    """Negamax over every reachable non-terminal position, side to move = +1.
    Returns {position: (value, [value of each move or None if illegal])}."""
    memo = {}

    def value(b):
        if b in memo:
            return memo[b][0]
        best, move_values = -2, [None] * 9
        for a in range(9):
            if b[a] != 0:
                continue
            nb = list(b)
            nb[a] = 1
            if _has_line(nb, 1):
                v = 1
            elif 0 not in nb:
                v = 0
            else:
                v = -value(tuple(-x for x in nb))
            move_values[a] = v
            best = max(best, v)
        memo[b] = (best, move_values)
        return best

    value((0,) * 9)
    return memo


def build_eval_data(device):
    memo = solve_tictactoe()
    pow3 = 3 ** np.arange(9)
    opt_table = np.zeros((3 ** 9, 9), dtype=bool)
    states, optimal = [], []
    for b, (best, move_values) in memo.items():
        mask = np.array([v is not None and v == best for v in move_values])
        states.append(b)
        optimal.append(mask)
        opt_table[int(((np.array(b) + 1) * pow3).sum())] = mask
    states = np.array(states, dtype=np.int8)
    optimal = np.array(optimal)
    legal = states == 0
    critical = (optimal != legal).any(1)  # at least one legal move throws the result away
    return {
        "states": torch.tensor(states, device=device),
        "legal": torch.tensor(legal, device=device),
        "optimal": torch.tensor(optimal, device=device),
        "critical": torch.tensor(critical, device=device),
        "opt_table": torch.tensor(opt_table, device=device),
        "pow3": torch.tensor(pow3, dtype=torch.long, device=device),
        "lines": torch.tensor(LINES, dtype=torch.long, device=device),
        "n_states": len(states),
        "n_critical": int(critical.sum()),
    }


# ----------------------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------------------

class DuelingQNet(nn.Module):
    def __init__(self, hidden=512, head_hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(27, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
        self.adv = nn.Sequential(nn.Linear(hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 9))

    def forward(self, x):
        h = self.body(x)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=1, keepdim=True)


def q_values(net, boards, amp):
    with autocast_ctx(amp):
        return net(encode(boards)).float()


def export_npz(net, path):
    """Plain NumPy weights for play_dqn.py: <group>_<i>_w has shape (in, out)."""
    arrays = {}
    for group in ("body", "value", "adv"):
        linears = [m for m in getattr(net, group) if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linears):
            arrays[f"{group}_{i}_w"] = layer.weight.detach().float().cpu().numpy().T.copy()
            arrays[f"{group}_{i}_b"] = layer.bias.detach().float().cpu().numpy().copy()
    np.savez(path, **arrays)


# ----------------------------------------------------------------------------------------
# GPU replay buffer and vectorised self-play
# ----------------------------------------------------------------------------------------

class GPUReplayBuffer:
    def __init__(self, capacity, chunk, device):
        capacity = max(chunk, capacity // chunk * chunk)  # whole chunks -> plain slice writes
        self.s = torch.zeros(capacity, 9, dtype=torch.int8, device=device)
        self.s2 = torch.zeros(capacity, 9, dtype=torch.int8, device=device)
        self.a = torch.zeros(capacity, dtype=torch.long, device=device)
        self.r = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.d = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.capacity, self.ptr, self.size, self.device = capacity, 0, 0, device

    def add(self, s, a, r, s2, d):
        n, p = s.shape[0], self.ptr
        self.s[p:p + n] = s
        self.a[p:p + n] = a
        self.r[p:p + n] = r
        self.s2[p:p + n] = s2
        self.d[p:p + n] = d
        self.ptr = (p + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch, perms, inv):
        """Uniform sample with a random board symmetry applied to every transition."""
        i = torch.randint(0, self.size, (batch,), device=self.device)
        k = torch.randint(0, 8, (batch,), device=self.device)
        p = perms[k]
        return (self.s[i].gather(1, p), inv[k, self.a[i]], self.r[i],
                self.s2[i].gather(1, p), self.d[i])


class SelfPlayEnvs:
    """n independent games stored as canonical boards (side to move = +1)."""

    def __init__(self, n, lines, device):
        self.boards = torch.zeros(n, 9, dtype=torch.int8, device=device)
        self.ply = torch.zeros(n, dtype=torch.long, device=device)
        self.ar = torch.arange(n, device=device)
        self.lines = lines
        self.stats = torch.zeros(3, dtype=torch.long, device=device)  # X wins, O wins, draws

    @torch.no_grad()
    def step(self, net, eps, buffer, amp):
        b = self.boards
        legal = b == 0
        greedy = q_values(net, b, amp).masked_fill(~legal, float("-inf")).argmax(1)
        random_a = torch.rand(b.shape, device=b.device).masked_fill(~legal, -1.0).argmax(1)
        a = torch.where(torch.rand(b.shape[0], device=b.device) < eps, random_a, greedy)

        new = b.clone()
        new[self.ar, a] = 1
        win = is_win(new, self.lines)
        draw = ~win & (new != 0).all(1)
        done = win | draw
        nxt = -new  # opponent's point of view
        buffer.add(b, a, win.float(), nxt, done)

        x_moved = self.ply % 2 == 0
        self.stats += torch.stack([(win & x_moved).sum(), (win & ~x_moved).sum(), draw.sum()])
        self.boards = torch.where(done[:, None], torch.zeros_like(nxt), nxt)
        self.ply = torch.where(done, torch.zeros_like(self.ply), self.ply + 1)


# ----------------------------------------------------------------------------------------
# Training step
# ----------------------------------------------------------------------------------------

def train_step(online, target, opt, scaler, buffer, perms, inv, cfg, amp):
    s, a, r, s2, d = buffer.sample(cfg["batch_size"], perms, inv)

    with torch.no_grad():
        with autocast_ctx(amp):
            x2 = encode(s2)
            q2_online = online(x2).float()
            q2_target = target(x2).float()
        a2 = q2_online.masked_fill(s2 != 0, float("-inf")).argmax(1, keepdim=True)
        y = torch.where(d, r, -cfg["gamma"] * q2_target.gather(1, a2).squeeze(1))

    with autocast_ctx(amp):
        q = online(encode(s)).float()
    q_sa = q.gather(1, a[:, None]).squeeze(1)
    loss = F.smooth_l1_loss(q_sa, y)

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(online.parameters(), cfg["grad_clip"])
    scaler.step(opt)
    scaler.update()
    return loss.detach()


@torch.no_grad()
def soft_update(target_params, online_params, tau):
    if hasattr(torch, "_foreach_lerp_"):
        torch._foreach_lerp_(target_params, online_params, tau)
    else:
        for pt, po in zip(target_params, online_params):
            pt.lerp_(po, tau)


def make_optimiser(params, lr, device):
    params = list(params)
    if device.type == "cuda":
        try:
            return torch.optim.Adam(params, lr=lr, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(params, lr=lr)


def make_grad_scaler(amp):
    try:
        return torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp)


# ----------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------

@torch.no_grad()
def position_accuracy(net, ev, amp):
    q = q_values(net, ev["states"], amp).masked_fill(~ev["legal"], float("-inf"))
    correct = ev["optimal"].gather(1, q.argmax(1, keepdim=True)).squeeze(1)
    n_ok = int(correct.sum().item())
    crit = correct[ev["critical"]].float().mean().item()
    return n_ok / ev["n_states"], crit, n_ok


@torch.no_grad()
def play_match(net, n_games, agent_first, opponent, ev, amp):
    """Greedy agent vs 'perfect' (random choice among minimax-optimal moves) or 'random'.
    Returns (wins, draws, losses) from the agent's point of view."""
    dev = ev["lines"].device
    board = torch.zeros(n_games, 9, dtype=torch.int8, device=dev)
    agent_turn = torch.full((n_games,), agent_first, dtype=torch.bool, device=dev)
    active = torch.ones(n_games, dtype=torch.bool, device=dev)
    result = torch.zeros(n_games, dtype=torch.long, device=dev)
    ar = torch.arange(n_games, device=dev)
    for _ in range(9):
        legal = board == 0
        agent_a = q_values(net, board, amp).masked_fill(~legal, float("-inf")).argmax(1)
        if opponent == "perfect":
            cand = ev["opt_table"][((board.long() + 1) * ev["pow3"]).sum(1)] & legal
        else:
            cand = legal
        opp_a = torch.rand(n_games, 9, device=dev).masked_fill(~cand, -1.0).argmax(1)
        a = torch.where(agent_turn, agent_a, opp_a)

        new = board.clone()
        new[ar, a] = 1
        win = is_win(new, ev["lines"])
        full = (new != 0).all(1)
        result = torch.where(active & win, agent_turn.long() * 2 - 1, result)
        active = active & ~(win | full)
        board = torch.where(active[:, None], -new, board)
        agent_turn = ~agent_turn
    return int((result == 1).sum()), int((result == 0).sum()), int((result == -1).sum())


MATCHES = [("perfect", True, "vs perfect as X"), ("perfect", False, "vs perfect as O"),
           ("random", True, "vs random as X"), ("random", False, "vs random as O")]


def evaluate(net, ev, n_games):
    # Always FP32, so the scores match play_dqn.py move for move
    acc, crit, n_ok = position_accuracy(net, ev, amp=False)
    m = {"acc": acc, "crit_acc": crit, "n_correct": n_ok}
    for opponent, first, name in MATCHES:
        m[name] = play_match(net, n_games, first, opponent, ev, amp=False)
    m["losses"] = sum(m[name][2] for _, _, name in MATCHES)
    return m


def wdl(t):
    return f"{t[0]}/{t[1]}/{t[2]}"


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------

def train(cfg):
    try:
        sys.stdout.reconfigure(line_buffering=True)  # live logs under `!python` on Colab
    except AttributeError:
        pass
    set_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"[{clock()}] GPU: {props.name} | {props.total_memory / 1024 ** 3:.1f} GB | "
              f"{props.multi_processor_count} SMs | PyTorch {torch.__version__}")
    elif cfg["allow_cpu"]:
        device = torch.device("cpu")
        print(f"[{clock()}] WARNING: no GPU found, training on CPU.")
    else:
        raise RuntimeError("No GPU found. In Colab: Runtime -> Change runtime type -> T4 GPU, "
                           "then run again (or set CONFIG['allow_cpu'] = True).")
    amp = cfg["use_amp"] and device.type == "cuda"

    t0 = time.time()
    ev = build_eval_data(device)
    print(f"[{clock()}] Minimax solved: {ev['n_states']} reachable positions, "
          f"{ev['n_critical']} critical ({time.time() - t0:.1f}s)")

    perms_np, inv_np = board_symmetries()
    perms = torch.tensor(perms_np, dtype=torch.long, device=device)
    inv = torch.tensor(inv_np, dtype=torch.long, device=device)

    online = DuelingQNet(cfg["hidden"], cfg["head_hidden"]).to(device)
    target = DuelingQNet(cfg["hidden"], cfg["head_hidden"]).to(device)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    target_params, online_params = list(target.parameters()), list(online.parameters())
    opt = make_optimiser(online.parameters(), cfg["lr"], device)
    scaler = make_grad_scaler(amp)

    n_envs = cfg["n_envs"]
    envs = SelfPlayEnvs(n_envs, ev["lines"], device)
    buffer = GPUReplayBuffer(cfg["buffer_size"], n_envs, device)
    eps_floor = torch.linspace(cfg["eps_floor_min"], cfg["eps_floor_max"], n_envs, device=device)

    iterations = cfg["iterations"]
    warmup_iters = math.ceil(cfg["warmup_transitions"] / n_envs)
    total_updates = max(1, (iterations - warmup_iters + 1) * cfg["updates_per_iter"])
    n_params = sum(p.numel() for p in online.parameters())
    print(f"[{clock()}] Network: {n_params:,} parameters | {n_envs} parallel games | "
          f"buffer {buffer.capacity:,} | batch {cfg['batch_size']} | AMP {'on' if amp else 'off'}")
    print(f"[{clock()}] Training for up to {iterations:,} iterations "
          f"({iterations * n_envs / 1e6:.0f}M moves, {total_updates:,} gradient updates)\n")
    print("columns: acc = optimal-move accuracy on all positions (critical positions in brackets),"
          " results are W/D/L for the model\n")

    pt_path = os.path.join(cfg["out_dir"], "tictactoe_dqn.pt")
    npz_path = os.path.join(cfg["out_dir"], "tictactoe_dqn.npz")
    best_key, best_metrics, perfect_streak = None, None, 0
    history = []
    n_updates, loss_sum, loss_count = 0, torch.zeros((), device=device), 0
    t_start = t_last = time.time()
    updates_last = 0

    for it in range(1, iterations + 1):
        eps_sched = max(0.0, cfg["eps_start"] * (1.0 - it / (cfg["eps_decay_frac"] * iterations)))
        eps = torch.clamp(eps_floor, min=eps_sched)
        envs.step(online, eps, buffer, amp)

        if buffer.size >= cfg["warmup_transitions"]:
            for _ in range(cfg["updates_per_iter"]):
                progress = min(1.0, n_updates / total_updates)
                lr = cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + math.cos(math.pi * progress))
                for g in opt.param_groups:
                    g["lr"] = lr
                loss_sum += train_step(online, target, opt, scaler, buffer, perms, inv, cfg, amp)
                soft_update(target_params, online_params, cfg["tau"])
                loss_count += 1
                n_updates += 1

        if it % cfg["eval_every"] != 0 and it != iterations:
            continue

        m = evaluate(online, ev, cfg["eval_games"])
        now = time.time()
        loss = (loss_sum / max(1, loss_count)).item()
        x_wins, o_wins, draws = envs.stats.tolist()
        games = max(1, x_wins + o_wins + draws)
        ups = (n_updates - updates_last) / max(1e-9, now - t_last)
        t_last, updates_last = now, n_updates
        loss_sum.zero_()
        loss_count = 0
        envs.stats.zero_()

        key = (m["n_correct"], -m["losses"])
        improved = best_key is None or key >= best_key
        if improved:
            best_key, best_metrics = key, dict(m, iteration=it)
            torch.save({"state_dict": online.state_dict(), "config": cfg, "metrics": best_metrics,
                        "iteration": it}, pt_path)
            export_npz(online, npz_path)

        history.append({"iteration": it, "updates": n_updates, "loss": loss, "acc": m["acc"],
                        "crit_acc": m["crit_acc"], "losses": m["losses"], "eps": eps_sched,
                        "selfplay_draw_rate": draws / games})
        print(f"[{clock()}] it {it:>6}/{iterations} | upd {n_updates:>6} | loss {loss:.5f} | "
              f"acc {100 * m['acc']:6.2f}% ({100 * m['crit_acc']:6.2f}%) | "
              f"perfect X {wdl(m['vs perfect as X'])} O {wdl(m['vs perfect as O'])} | "
              f"random X {wdl(m['vs random as X'])} O {wdl(m['vs random as O'])} | "
              f"self-play draws {100 * draws / games:4.1f}% | {ups:5.0f} upd/s | "
              f"{fmt_duration(now - t_start)}{'  * saved' if improved else ''}")

        perfect = m["n_correct"] == ev["n_states"] and m["losses"] == 0
        perfect_streak = perfect_streak + 1 if perfect else 0
        if (perfect_streak >= cfg["early_stop_evals"]
                and it >= cfg["early_stop_min_frac"] * iterations):
            print(f"\n[{clock()}] Perfect play held for {perfect_streak} evaluations in a row - stopping early.")
            break

    train_time = time.time() - t_start

    # ------------------------------------------------------------------ final report
    ckpt = torch.load(pt_path, map_location=device)
    online.load_state_dict(ckpt["state_dict"])
    final = evaluate(online, ev, cfg["final_eval_games"])
    with open(os.path.join(cfg["out_dir"], "history.json"), "w") as f:
        json.dump({"config": cfg, "history": history, "best_iteration": ckpt["iteration"],
                   "final": {k: (list(v) if isinstance(v, tuple) else v) for k, v in final.items()}},
                  f, indent=2)
    plot_history(history, os.path.join(cfg["out_dir"], "training_curves.png"))

    print("\n" + "=" * 72)
    print(f" BEST MODEL (iteration {ckpt['iteration']:,})   training time {fmt_duration(train_time)}")
    print("=" * 72)
    print(f" Optimal-move accuracy, all positions : {100 * final['acc']:.2f}%  "
          f"({final['n_correct']}/{ev['n_states']})")
    print(f" Optimal-move accuracy, critical ones : {100 * final['crit_acc']:.2f}%  ({ev['n_critical']} positions)")
    for _, _, name in MATCHES:
        w, d, l = final[name]
        n = w + d + l
        print(f" {name:<17}: win {100 * w / n:6.2f}%  draw {100 * d / n:6.2f}%  "
              f"loss {100 * l / n:6.2f}%   ({n:,} games)")
    print("=" * 72)
    print(f" Saved: {npz_path}\n        {pt_path}")

    if cfg["save_to_drive"] and IN_COLAB:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            dst = "/content/drive/MyDrive/tictactoe_dqn"
            shutil.copytree(cfg["out_dir"], dst, dirs_exist_ok=True)
            print(f" Copied to Google Drive: {dst}")
        except Exception as e:
            print(f" Google Drive copy skipped: {e}")
    if cfg["download_at_end"] and IN_COLAB and in_notebook():
        try:
            from google.colab import files
            files.download(npz_path)
        except Exception as e:
            print(f" Automatic download skipped: {e}")
    return final


def plot_history(history, path):
    try:
        import matplotlib
        if not in_notebook():
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    its = [h["iteration"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(its, [100 * h["acc"] for h in history], label="all positions")
    ax1.plot(its, [100 * h["crit_acc"] for h in history], label="critical positions")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("optimal-move accuracy (%)")
    ax1.set_title("Accuracy vs exact minimax")
    ax1.grid(alpha=0.3)
    ax1.legend()
    trained = [h for h in history if h["updates"] > 0 and h["loss"] > 0]
    if trained:
        ax2.plot([h["iteration"] for h in trained], [h["loss"] for h in trained])
        ax2.set_yscale("log")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("Huber loss")
    ax2.set_title("TD loss")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    if in_notebook():
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    train(CONFIG)
