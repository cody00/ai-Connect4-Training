Connect 4 — Self‑Play Learning + Strong Alpha‑Beta Search (Pure Python)

Overview

This project implements a Connect 4 engine that learns by self‑play and plays using a fast alpha‑beta search over bitboards — all in pure Python with no external dependencies.

Highlights
- Bitboards + alpha‑beta with iterative deepening, PVS, transposition table, killer moves, center‑first ordering, null‑move pruning, and late‑move reductions.
- Self‑play training on a compact, hand‑crafted feature vector; the value function is linear and fast.
- Training and gameplay are aligned: both use the same bitboard features, and the learned value estimates f_my − f_opp.
- CLI for train/eval/play and a small Tkinter GUI for clickable play.

Quick Start
- Train (fresh or resume): `python connect4_ai.py train --episodes 50000 --model model.json`
- Train using the search policy (recommended):
  - Depth‑limited: `python connect4_ai.py train --episodes 50000 --model model.json --train-depth 4`
  - Time‑based: `python connect4_ai.py train --episodes 50000 --model model.json --train-time-ms 100`
- Evaluate vs random: `python connect4_ai.py eval --episodes 1000 --model model.json --depth 6`
- Play in terminal: `python connect4_ai.py play --model model.json --time-ms 200`
- Play with GUI: `python connect4_gui.py --model model.json --time-ms 200` (add `--ai-first` if you want the AI to start)

How It Learns
- Self‑Play Episodes: Two agents with shared weights alternate moves until the game ends. During training, exploration is controlled by ε‑greedy at the root.
- Policies for Training:
  - Q‑greedy: Chooses the move that maximizes the learned value on the next state (with ε chance of a random move).
  - Search policy: If `--train-depth` or `--train-time-ms` is provided, moves are chosen by the same alpha‑beta search used in play/eval (with ε randomization at the root). This typically produces stronger training trajectories.
- Value Function: A linear model `V(s, player) = dot(w, f_my(s) − f_opp(s))` where `f_*` are bitboard features described below.
- Targets and Updates: After each episode, every move receives a Monte‑Carlo target: +1 for the winner’s moves, −1 for the loser’s moves, 0 for draws, discounted by `gamma` based on distance from the end. The gradient is the feature difference vector.
- Epsilon Schedule: `--eps-start` decays multiplicatively by `--eps-decay` down to `--eps-end` across episodes (default: 1.0 → 0.05).
- Persistence: The trainer tries to load `--model` at start; if found, it continues learning from existing weights. It saves periodically and at the end.

How Gameplay Works
- Search: The engine uses bitboards and a negamax alpha‑beta search with iterative deepening to pick a move.
- Evaluation: The same learned linear features used for training are used inside the search (feature‑difference form), ensuring learning and play are aligned.
- Time vs Depth: You can either fix a depth via `--depth` or give a time budget per move via `--time-ms`. Time‑based search adapts depth to your machine.

Algorithms & Engine Internals
- Board Representations:
  - 2D Board: Simple 6×7 grid for training loops and GUI updates.
  - Bitboards: Two 64‑bit bitboards (one per player) with a 7‑bit stride per column, enabling very fast move generation and win detection.
- Feature Engineering (per player):
  - `bias`: constant 1.0
  - `center`: fraction of the player’s stones in the center column
  - Window counts across all 4‑in‑a‑row windows: `my1`, `my2`, `my3` for the player; `opp1`, `opp2`, `opp3` for the opponent, considering only windows that are still “open” (i.e., have empties and no opponent stones for the player’s counts, and vice‑versa).
  - `win_now`: 1.0 if the player already has a 4‑in‑a‑row, else 0.0
  - Features are normalized (counts divided by total number of windows) and combined as `f_my − f_opp` for evaluation.
- Search Details:
  - Negamax alpha‑beta with Principal Variation Search (PVS) on non‑first children and late‑move reductions on non‑killer moves.
  - Null‑move pruning at sufficient depth to accelerate cutoffs.
  - Transposition Table keyed by `(p1_bitboard, p2_bitboard, player)` storing depth/score/bounds and preferred move.
  - Move Ordering: Prefer PV/TT move, then killer moves (per ply), then center‑first column bias.
  - Tactical Heuristics: Immediate win detection; urgent single‑threat block; double‑threat detection that returns a forcing move when available.
  - Time Management: Iterative deepening up to `--depth`, or stop when the time budget `--time-ms` elapses.

Illustrations
- Board Coordinates (rows 0–5 top→bottom, columns 0–6 left→right)

  0 1 2 3 4 5 6
  . . . . . . .   row 0
  . . . . . . .   row 1
  . . . . . . .   row 2
  . . . . . . .   row 3
  . . . . . . .   row 4
  . . . . . . .   row 5

- Bitboard Layout (7‑bit stride per column): bit index = `c*7 + r`

  Column 0 bit indices (top→bottom):
  r=0→bit 0, r=1→bit 1, …, r=5→bit 5

  Columns step by +7 in bit index:
  c=0 base=0, c=1 base=7, c=2 base=14, …, c=6 base=42

- Gravity trick (per column):
  `move = (mask + bottom_mask_col) & col_mask` chooses the lowest empty bit in that column in O(1),
  where `mask = p1 | p2`, `col_mask = (((1<<6)−1) << c*7)`, and `bottom_mask_col = 1 << (c*7)`.

- Window Types (4 in a row):
  - Horizontal: (r, c) … (r, c+3)
  - Vertical:   (r, c) … (r+3, c)
  - Diagonal \\ : (r, c) … (r+3, c+3)
  - Diagonal / : (r, c) … (r−3, c+3)

  Example (a diagonal \\ window marked with X):

  . . . X . . .
  . . X . . . .
  . X . . . . .
  X . . . . . .
  . . . . . . .
  . . . . . . .

CLI Usage
- Train
  - `python connect4_ai.py train [options]`
  - Options:
    - `--episodes INT`: number of self‑play games (default 20000)
    - `--alpha FLOAT`: learning rate (default 0.01)
    - `--gamma FLOAT`: discount factor for distance‑from‑end (default 0.98)
    - `--eps-start FLOAT`: starting ε (default 1.0)
    - `--eps-end FLOAT`: minimum ε (default 0.05)
    - `--eps-decay FLOAT`: multiplicative decay per episode (default 0.999)
    - `--seed INT`: random seed
    - `--model PATH`: JSON file to load/save weights (default `model.json`)
    - `--train-depth INT`: use depth‑limited alpha‑beta to pick moves during training (with ε exploration)
    - `--train-time-ms INT`: use time‑limited alpha‑beta to pick moves during training (overrides depth)
- Evaluate vs Random
  - `python connect4_ai.py eval --episodes 1000 --model model.json [--depth 6 | --time-ms 200]`
  - Runs games as both first and second player; reports W/D/L and rates.
- Play (Terminal)
  - `python connect4_ai.py play --model model.json [--depth 6 | --time-ms 200]`
  - Enter a column number 0–6 each turn; `q` to quit.
- Play (GUI)
  - `python connect4_gui.py --model model.json [--depth 6 | --time-ms 200] [--ai-first]`
  - Click a column to drop your piece (Red). Toggle first mover and press “New Game” to restart.

Training Tips
- Start Simple: If just testing, run a few thousand episodes with `--train-depth 3` or `--train-time-ms 50–150` and moderate exploration (e.g., `--eps-start 0.3` → `--eps-end 0.05`).
- Time‑Based Search: Prefer `--time-ms` for training policy and gameplay to adapt to different CPUs without changing depth.
- Resuming: Reuse the same `--model` path to continue learning; to start fresh, point to a new file or delete the old one.
- Seeds: Use `--seed` for reproducibility when comparing settings.

Model Persistence
- On `train`, the script first attempts to load `--model` and continues learning from those weights; if not found, it starts a new model.
- The trainer saves periodically and at the end to the same file (by default, every 5000 episodes and after the last episode).

FAQ
- Does training start over each time? No. If `--model` exists, training resumes from it; otherwise a fresh model is created.
- Is self‑play just random? Early training uses ε‑greedy exploration; as ε decays, choices become guided by the learned value/search. With `--train-depth` or `--train-time-ms`, move selection is search‑based with a small ε for exploration.
- Depth vs time for play? Time‑based (`--time-ms`) is recommended for more consistent move quality across machines.

Performance Notes
- Pure Python. Bitboards and tight inner loops keep it responsive up to moderate depths; time‑based search helps avoid stalls on slower CPUs.

Future Ideas
- Temporal‑difference updates (TD(0)/n‑step) for faster credit assignment.
- Opponent diversity (e.g., self‑play against snapshots of older models).
- Softmax/Boltzmann exploration over move values.
- Symmetry augmentation by mirroring columns during training.
