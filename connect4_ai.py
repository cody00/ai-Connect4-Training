#!/usr/bin/env python3
"""
Connect 4 with self-play learning and human vs AI mode.

No third party packages are required. Pure Python.

Training uses a very simple linear value function over hand crafted features.
It improves with many self play games. You can train, evaluate, and play.
For gameplay the AI now uses depth-limited alpha-beta search with good move
ordering, which makes it significantly stronger while still remaining fast
and pure Python.

Usage examples:

  Train a model for fifty thousand games and save it
    python connect4_ai.py train --episodes 50000 --model model.json

  Evaluate the trained model against a random player
    python connect4_ai.py eval --episodes 1000 --model model.json

  Play against the trained model
    python connect4_ai.py play --model model.json

  See all options
    python connect4_ai.py -h
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

ROWS = 6
COLS = 7
CONNECT = 4

# Precompute all 4-cell windows on the board for fast feature evaluation
WINDOWS: List[List[Tuple[int, int]]] = []

# Horizontal windows
for r in range(ROWS):
    for c in range(COLS - CONNECT + 1):
        WINDOWS.append([(r, c + i) for i in range(CONNECT)])
# Vertical windows
for r in range(ROWS - CONNECT + 1):
    for c in range(COLS):
        WINDOWS.append([(r + i, c) for i in range(CONNECT)])
# Diagonal down-right
for r in range(ROWS - CONNECT + 1):
    for c in range(COLS - CONNECT + 1):
        WINDOWS.append([(r + i, c + i) for i in range(CONNECT)])
# Diagonal up-right
for r in range(CONNECT - 1, ROWS):
    for c in range(COLS - CONNECT + 1):
        WINDOWS.append([(r - i, c + i) for i in range(CONNECT)])

N_WINDOWS = len(WINDOWS)  # should be 69 for a 6x7 board


Board = List[List[int]]  # 0 empty, 1 player one, -1 player two

# Default search depth for the alpha-beta AI. You can override via CLI.
DEFAULT_SEARCH_DEPTH = 5

# Large win/loss sentinel values for search scoring
WIN_SCORE = 1_000_000.0
LOSS_SCORE = -WIN_SCORE


class SearchTimeout(Exception):
    pass


# --- Bitboard constants & helpers (7-bit stride per column) ---

STRIDE = ROWS + 1  # 7
BOTTOM_MASK_COL = [1 << (c * STRIDE) for c in range(COLS)]
COL_MASK = [(((1 << ROWS) - 1) << (c * STRIDE)) for c in range(COLS)]
TOP_PLAY_MASK_COL = [1 << (c * STRIDE + ROWS - 1) for c in range(COLS)]

# Bit masks for each 4-cell window for fast counting on bitboards
BIT_WINDOWS: List[int] = []
for window in WINDOWS:
    m = 0
    for (r, c) in window:
        m |= (1 << (c * STRIDE + r))
    BIT_WINDOWS.append(m)


def bb_from_board(board: Board) -> Tuple[int, int]:
    """Convert 2D board (values -1,0,1) into two bitboards (p1, p2)."""
    p1 = 0
    p2 = 0
    for r in range(ROWS):
        for c in range(COLS):
            v = board[r][c]
            if v:
                idx = c * STRIDE + r
                if v == 1:
                    p1 |= (1 << idx)
                else:
                    p2 |= (1 << idx)
    return p1, p2


def bb_valid_moves(mask: int) -> List[int]:
    return [c for c in range(COLS) if (mask & TOP_PLAY_MASK_COL[c]) == 0]


def bb_make_move(p1: int, p2: int, col: int, player: int) -> Tuple[int, int]:
    """Return updated (p1, p2) after current player drops in col."""
    mask = p1 | p2
    move = (mask + BOTTOM_MASK_COL[col]) & COL_MASK[col]
    if move == 0:
        return p1, p2  # column full; no change
    if player == 1:
        p1 |= move
    else:
        p2 |= move
    return p1, p2


def bb_has_won(pos: int) -> bool:
    """Bitboard 4-in-a-row check using shifts in 7x6 with 7-bit stride."""
    # Vertical (shift 1)
    m = pos & (pos >> 1)
    if (m & (m >> 2)) != 0:
        return True
    # Horizontal (shift 7)
    m = pos & (pos >> STRIDE)
    if (m & (m >> (2 * STRIDE))) != 0:
        return True
    # Diagonal / (shift 6)
    m = pos & (pos >> (STRIDE - 1))
    if (m & (m >> (2 * (STRIDE - 1)))) != 0:
        return True
    # Diagonal \ (shift 8)
    m = pos & (pos >> (STRIDE + 1))
    if (m & (m >> (2 * (STRIDE + 1)))) != 0:
        return True
    return False


def bb_features(my_p: int, opp_p: int) -> List[float]:
    """Compute features for bitboards from the perspective of `my_p`."""
    center_col = COLS // 2
    center_bits = my_p & COL_MASK[center_col]
    center_count = center_bits.bit_count()

    my1 = my2 = my3 = 0
    opp1 = opp2 = opp3 = 0
    for mask in BIT_WINDOWS:
        my = (my_p & mask).bit_count()
        opp = (opp_p & mask).bit_count()
        empty = 4 - (my + opp)
        if opp == 0 and my > 0 and empty > 0:
            if my == 1:
                my1 += 1
            elif my == 2:
                my2 += 1
            elif my == 3:
                my3 += 1
        if my == 0 and opp > 0 and empty > 0:
            if opp == 1:
                opp1 += 1
            elif opp == 2:
                opp2 += 1
            elif opp == 3:
                opp3 += 1
    win_now = 1.0 if bb_has_won(my_p) else 0.0
    f_bias = 1.0
    f_center = center_count / ROWS
    nf = float(N_WINDOWS)
    return [
        f_bias,
        f_center,
        my1 / nf,
        my2 / nf,
        my3 / nf,
        opp1 / nf,
        opp2 / nf,
        opp3 / nf,
        win_now,
    ]


def bb_evaluate_board(p1: int, p2: int, player: int, model: 'Model') -> float:
    my_p = p1 if player == 1 else p2
    opp_p = p2 if player == 1 else p1
    f_my = bb_features(my_p, opp_p)
    f_opp = bb_features(opp_p, my_p)
    return dot(model.weights, f_my) - dot(model.weights, f_opp)



def empty_board() -> Board:
    return [[0 for _ in range(COLS)] for _ in range(ROWS)]


def copy_board(board: Board) -> Board:
    return [row[:] for row in board]


def valid_moves(board: Board) -> List[int]:
    return [c for c in range(COLS) if board[0][c] == 0]


def drop_piece(board: Board, col: int, player: int) -> Optional[Tuple[int, int]]:
    """Drop a piece for player (1 or -1) into column col. Return (row, col) or None if column full."""
    if not (0 <= col < COLS) or board[0][col] != 0:
        return None
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == 0:
            board[r][col] = player
            return (r, col)
    return None


def check_winner(board: Board, player: int) -> bool:
    """Check if the given player has four in a row anywhere."""
    p = player
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if board[r][c] == p and board[r][c+1] == p and board[r][c+2] == p and board[r][c+3] == p:
                return True
    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if board[r][c] == p and board[r+1][c] == p and board[r+2][c] == p and board[r+3][c] == p:
                return True
    # Diagonal down-right
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if board[r][c] == p and board[r+1][c+1] == p and board[r+2][c+2] == p and board[r+3][c+3] == p:
                return True
    # Diagonal up-right
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if board[r][c] == p and board[r-1][c+1] == p and board[r-2][c+2] == p and board[r-3][c+3] == p:
                return True
    return False


def is_full(board: Board) -> bool:
    return all(board[0][c] != 0 for c in range(COLS))


def print_board(board: Board) -> None:
    """Pretty print board for play mode. X is player one, O is player two."""
    chars = {0: '.', 1: 'X', -1: 'O'}
    print()
    for r in range(ROWS):
        print(' '.join(chars[board[r][c]] for c in range(COLS)))
    print('0 1 2 3 4 5 6   enter a column number')
    print()


@dataclass
class Model:
    weights: List[float]

    @staticmethod
    def new():
        # Feature vector length: bias + center + my1 + my2 + my3 + opp1 + opp2 + opp3 + win
        return Model(weights=[0.0]*9)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({"weights": self.weights}, f)

    @staticmethod
    def load(path: str) -> "Model":
        with open(path, 'r') as f:
            data = json.load(f)
        return Model(weights=list(map(float, data["weights"])))


def evaluate_features(board: Board, player: int) -> List[float]:
    """Compute a small set of features for the current board from the perspective of player.
    Values are normalized to small ranges for stable learning.
    Returns a list of floats length 9.
    """
    # center control
    center_col = COLS // 2
    center_count = sum(1 for r in range(ROWS) if board[r][center_col] == player)
    # window counts
    my1 = my2 = my3 = 0
    opp1 = opp2 = opp3 = 0
    for window in WINDOWS:
        vals = [board[r][c] for (r, c) in window]
        my = vals.count(player)
        opp = vals.count(-player)
        empty = vals.count(0)
        if opp == 0 and my > 0 and empty > 0:
            if my == 1:
                my1 += 1
            elif my == 2:
                my2 += 1
            elif my == 3:
                my3 += 1
        if my == 0 and opp > 0 and empty > 0:
            if opp == 1:
                opp1 += 1
            elif opp == 2:
                opp2 += 1
            elif opp == 3:
                opp3 += 1
    # immediate win signal
    win_now = 1.0 if check_winner(board, player) else 0.0

    # Normalize features
    f_bias = 1.0
    f_center = center_count / ROWS  # in [0,1]
    nf = float(N_WINDOWS)
    features = [
        f_bias,
        f_center,
        my1 / nf,
        my2 / nf,
        my3 / nf,
        opp1 / nf,
        opp2 / nf,
        opp3 / nf,
        win_now,
    ]
    return features


def dot(w: List[float], f: List[float]) -> float:
    return sum(wi * fi for wi, fi in zip(w, f))


def q_value_after_move(board: Board, col: int, player: int, model: Model) -> Optional[float]:
    """Return Q(s,a) as the bitboard evaluation difference on s' (after the move).
    None if invalid move.
    """
    if board[0][col] != 0:
        return None
    b2 = copy_board(board)
    drop_piece(b2, col, player)
    p1, p2 = bb_from_board(b2)
    return bb_evaluate_board(p1, p2, player, model)


def choose_action(board: Board, player: int, model: Model, epsilon: float) -> int:
    moves = valid_moves(board)
    if not moves:
        return -1
    if random.random() < epsilon:
        return random.choice(moves)
    # Greedy choice with random tie break
    scored = []
    best = -float('inf')
    for c in moves:
        q = q_value_after_move(board, c, player, model)
        if q is None:
            continue
        scored.append((q, c))
        if q > best:
            best = q
    # Pick among the max q moves at random for variety
    best_moves = [c for q, c in scored if q == best]
    return random.choice(best_moves) if best_moves else random.choice(moves)


# --- Stronger gameplay AI: depth-limited negamax with alpha-beta pruning ---

def evaluate_board(board: Board, player: int, model: Model) -> float:
    """Heuristic evaluation: advantage for current player.
    Uses our linear features for both players and returns their difference.
    """
    my_feats = evaluate_features(board, player)
    opp_feats = evaluate_features(board, -player)
    return dot(model.weights, my_feats) - dot(model.weights, opp_feats)


def ordered_moves(board: Board, prefer: Optional[List[int]] = None, killers: Optional[List[int]] = None) -> List[int]:
    """Return valid moves ordered for better pruning.
    Priority: prefer (e.g., PV move), then killer moves, then center-first.
    Duplicates are removed while preserving priority order.
    """
    moves_set = set(valid_moves(board))
    if not moves_set:
        return []
    order_center = [3, 2, 4, 1, 5, 0, 6]
    seq: List[int] = []
    if prefer:
        for c in prefer:
            if c in moves_set and c not in seq:
                seq.append(c)
    if killers:
        for c in killers:
            if c in moves_set and c not in seq:
                seq.append(c)
    for c in order_center:
        if c in moves_set and c not in seq:
            seq.append(c)
    return seq


def ordered_moves_bb(mask: int, prefer: Optional[List[int]] = None, killers: Optional[List[int]] = None) -> List[int]:
    """Order valid moves from bitboard mask.
    Priority: prefer (TT/PV), killers, then center-first.
    """
    valid = {c for c in range(COLS) if (mask & TOP_PLAY_MASK_COL[c]) == 0}
    seq: List[int] = []
    if prefer:
        for c in prefer:
            if c in valid and c not in seq:
                seq.append(c)
    if killers:
        for c in killers:
            if c in valid and c not in seq:
                seq.append(c)
    for c in [3, 2, 4, 1, 5, 0, 6]:
        if c in valid and c not in seq:
            seq.append(c)
    return seq


def bb_immediate_winning_moves(p1: int, p2: int, player: int) -> List[int]:
    """Columns where current player wins immediately by dropping a piece now."""
    mask = p1 | p2
    wins: List[int] = []
    for c in range(COLS):
        if (mask & TOP_PLAY_MASK_COL[c]) != 0:
            continue
        move = (mask + BOTTOM_MASK_COL[c]) & COL_MASK[c]
        if move == 0:
            continue
        if player == 1:
            if bb_has_won(p1 | move):
                wins.append(c)
        else:
            if bb_has_won(p2 | move):
                wins.append(c)
    return wins


def board_key(board: Board, player: int) -> Tuple[int, Tuple[int, ...]]:
    """A hashable key for TT: (player, flattened board)."""
    flat = tuple(board[r][c] for r in range(ROWS) for c in range(COLS))
    return (player, flat)


def negamax_bb(p1: int,
               p2: int,
               depth: int,
               alpha: float,
               beta: float,
               player: int,
               model: Model,
               tt: Dict[Tuple[int, int, int], Tuple[int, float, int, int]],
               killers: Dict[int, List[int]],
               ply: int,
               start_time: float,
               time_limit: Optional[float]) -> Tuple[float, int]:
    """Negamax with alpha-beta pruning on bitboards, with PVS and null-move pruning."""
    # Time check
    if time_limit is not None and (time.perf_counter() - start_time) >= time_limit:
        raise SearchTimeout()

    mask = p1 | p2
    # Terminal: previous move by opponent wins
    opp_pos = p2 if player == 1 else p1
    if bb_has_won(opp_pos):
        stones = mask.bit_count()
        return LOSS_SCORE + stones, -1
    # Draw
    if not bb_valid_moves(mask):
        return 0.0, -1

    if depth == 0:
        return bb_evaluate_board(p1, p2, player, model), -1

    key = (p1, p2, player)
    if key in tt:
        stored_depth, stored_score, flag, stored_move = tt[key]
        if stored_depth >= depth:
            if flag == 0:
                return stored_score, stored_move
            elif flag == 1 and stored_score > alpha:
                alpha = stored_score
            elif flag == 2 and stored_score < beta:
                beta = stored_score
            if alpha >= beta:
                return stored_score, stored_move

    best_score = -float('inf')
    best_move = -1
    alpha_orig = alpha

    prefer = []
    if key in tt and tt[key][3] != -1:
        prefer.append(tt[key][3])
    km = killers.get(ply, [])
    moves = ordered_moves_bb(mask, prefer=prefer, killers=km)

    # Null-move pruning (conservative): skip move and see if opponent can still not avoid beta cutoff
    # Apply only at sufficient depth and when there is at least one legal move
    if depth >= 4 and moves:
        R = 2
        try:
            score_nm, _ = negamax_bb(p1, p2, depth - 1 - R, -beta, -beta + 1, -player, model, tt, killers, ply + 1, start_time, time_limit)
            score_nm = -score_nm
            if score_nm >= beta:
                return score_nm, -1
        except SearchTimeout:
            pass

    first = True
    for col in moves:
        if (mask & TOP_PLAY_MASK_COL[col]) != 0:
            continue
        # Compute move bit and apply
        move = (mask + BOTTOM_MASK_COL[col]) & COL_MASK[col]
        mask2 = mask | move
        if player == 1:
            p1n, p2n = p1 | move, p2
            just_pos = p1n
        else:
            p1n, p2n = p1, p2 | move
            just_pos = p2n
        # Immediate win
        if bb_has_won(just_pos):
            stones = mask2.bit_count()
            score = WIN_SCORE - stones
        else:
            try:
                if first:
                    # Full window on first child (PV node)
                    child_score, _ = negamax_bb(p1n, p2n, max(depth - 1, 0), -beta, -alpha, -player, model, tt, killers, ply + 1, start_time, time_limit)
                    score = -child_score
                    first = False
                else:
                    # Late-move reduction on non-PV moves
                    reduced = depth - 1
                    if depth >= 3 and col not in km:
                        reduced = depth - 2
                    # PVS: try narrow window with reduced depth
                    child_score, _ = negamax_bb(p1n, p2n, max(reduced, 0), -(alpha + 1e-9), -alpha, -player, model, tt, killers, ply + 1, start_time, time_limit)
                    score = -child_score
                    if score > alpha:
                        # Re-search with full window at full depth
                        child_score, _ = negamax_bb(p1n, p2n, max(depth - 1, 0), -beta, -alpha, -player, model, tt, killers, ply + 1, start_time, time_limit)
                        score = -child_score
            except SearchTimeout:
                raise

        if score > best_score:
            best_score = score
            best_move = col
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            if col not in killers.setdefault(ply, []):
                killers[ply].insert(0, col)
                killers[ply] = killers[ply][:2]
            break

    flag = 0  # EXACT
    if best_score <= alpha_orig:
        flag = 2  # UPPERBOUND
    elif best_score >= beta:
        flag = 1  # LOWERBOUND
    tt[key] = (depth, best_score, flag, best_move)
    return best_score, best_move


def find_best_move(board: Board, player: int, model: Model, max_depth: int, time_ms: Optional[int]) -> int:
    """Iterative deepening bitboard search up to max_depth or time limit (ms)."""
    p1, p2 = bb_from_board(board)
    mask = p1 | p2
    moves = bb_valid_moves(mask)
    if not moves:
        return -1
    # Immediate win
    wins_now = bb_immediate_winning_moves(p1, p2, player)
    if wins_now:
        return wins_now[0]

    # Urgent block: if opponent has a single immediate winning move, block it.
    opp_wins = bb_immediate_winning_moves(p1, p2, -player)
    if len(opp_wins) == 1 and opp_wins[0] in moves:
        return opp_wins[0]

    # Double-threat finder: if any move creates two distinct immediate wins next turn, it's a forced win.
    for col in ordered_moves_bb(mask):
        move = (mask + BOTTOM_MASK_COL[col]) & COL_MASK[col]
        if move == 0:
            continue
        if player == 1:
            p1n, p2n = p1 | move, p2
        else:
            p1n, p2n = p1, p2 | move
        # Count our immediate winning replies on next move
        next_wins = bb_immediate_winning_moves(p1n, p2n, player)
        if len(next_wins) >= 2:
            return col

    tt: Dict[Tuple[int, int, int], Tuple[int, float, int, int]] = {}
    killers: Dict[int, List[int]] = {}
    best_move = moves[0]
    time_limit = None if time_ms is None else (time_ms / 1000.0)
    start = time.perf_counter()

    for depth in range(1, max_depth + 1):
        try:
            score, move = negamax_bb(p1, p2, depth, -float('inf'), float('inf'), player, model, tt, killers, 0, start, time_limit)
            if move != -1:
                best_move = move
            if abs(score) >= WIN_SCORE - 1000:
                break
        except SearchTimeout:
            break
    return best_move


@dataclass
class TrainConfig:
    episodes: int = 10000
    alpha: float = 0.01
    gamma: float = 0.98
    eps_start: float = 1.00
    eps_end: float = 0.05
    eps_decay: float = 0.999  # multiplicative decay per episode
    seed: Optional[int] = None
    # Optional: use search policy during self-play
    train_depth: Optional[int] = None
    train_time_ms: Optional[int] = None


def bb_feature_diff(p1: int, p2: int, player: int) -> List[float]:
    """Return (f_my - f_opp) using bitboard features from the player's perspective."""
    my_p = p1 if player == 1 else p2
    opp_p = p2 if player == 1 else p1
    f_my = bb_features(my_p, opp_p)
    f_opp = bb_features(opp_p, my_p)
    return [a - b for a, b in zip(f_my, f_opp)]


def play_episode_self_play(model: Model, cfg: TrainConfig) -> Tuple[int, List[Tuple[List[float], int]]]:
    """Run one self-play game.
    Returns winner (1, -1, 0) and history of (feature_diff_after_move, player).
    If cfg.train_depth/time_ms are provided, uses search policy for action selection
    with epsilon chance to pick a random legal move.
    """
    board = empty_board()
    player = 1  # player one starts
    history: List[Tuple[List[float], int]] = []
    # Epsilon can be scheduled per move but here we pass it in via cfg and handle outside
    while True:
        # choose action
        moves = valid_moves(board)
        if not moves:
            winner = 0
            break
        if cfg.train_depth is not None or cfg.train_time_ms is not None:
            # Search policy with exploration at the root
            if random.random() < cfg.eps_start:
                col = random.choice(moves)
            else:
                depth = cfg.train_depth if cfg.train_depth is not None else DEFAULT_SEARCH_DEPTH
                col = ai_move(board, player, model, depth=depth, time_ms=cfg.train_time_ms)
                if col not in moves:
                    # Fallback safeguard
                    col = random.choice(moves)
        else:
            # Q-greedy policy (uses bitboard evaluation difference under the hood)
            col = choose_action(board, player, model, epsilon=cfg.eps_start)
        # apply
        pos = drop_piece(board, col, player)
        # record (feature_diff) after move for this player using bitboards
        p1, p2 = bb_from_board(board)
        dfeats = bb_feature_diff(p1, p2, player)
        history.append((dfeats, player))
        # check end
        if check_winner(board, player):
            winner = player
            break
        if is_full(board):
            winner = 0
            break
        player *= -1
    return winner, history


def update_model_from_episode(model: Model, history: List[Tuple[List[float], int]], winner: int, alpha: float, gamma: float) -> None:
    """Monte Carlo style update on feature-difference vectors.
    Target: +1 for winner moves, -1 for loser moves, 0 for draws, discounted by distance from end.
    Uses V(s) = dot(w, f_my - f_opp) and gradient (f_my - f_opp).
    """
    # Walk from the end to give a little more credit to later moves
    T = len(history)
    for t, (feats, p) in enumerate(history):
        steps_from_end = T - t - 1
        discount = (gamma ** steps_from_end)
        if winner == 0:
            target = 0.0
        else:
            target = 1.0 if p == winner else -1.0
        target *= discount
        prediction = dot(model.weights, feats)
        delta = target - prediction
        # gradient is feats since Q is linear in weights
        for i in range(len(model.weights)):
            model.weights[i] += alpha * delta * feats[i]


def train(model: Model, cfg: TrainConfig, save_path: Optional[str] = None, log_every: int = 500) -> None:
    rng = random.Random(cfg.seed)
    random.seed(cfg.seed)

    eps = cfg.eps_start
    wins = draws = losses = 0
    start = time.time()
    for ep in range(1, cfg.episodes + 1):
        # For exploration we pass the current epsilon via cfg.eps_start each episode
        game_cfg = TrainConfig(
            eps_start=eps,
            train_depth=cfg.train_depth,
            train_time_ms=cfg.train_time_ms,
        )
        winner, hist = play_episode_self_play(model, game_cfg)
        update_model_from_episode(model, hist, winner, cfg.alpha, cfg.gamma)
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
        # decay epsilon
        eps = max(cfg.eps_end, eps * cfg.eps_decay)
        # logs
        if ep % log_every == 0 or ep == 1:
            rate = ep / max(1.0, time.time() - start)
            print(f"Episode {ep:6d} | W {wins:5d} D {draws:5d} L {losses:5d} | eps {eps:.3f} | {rate:.1f} games per sec")
            wins = draws = losses = 0
        if save_path and ep % (log_every * 10) == 0:
            Model(weights=model.weights).save(save_path)

    if save_path:
        Model(weights=model.weights).save(save_path)
        print(f"Saved model to {save_path}")


def ai_move(board: Board, player: int, model: Model, depth: int = DEFAULT_SEARCH_DEPTH, time_ms: Optional[int] = None) -> int:
    # Use iterative deepening + alpha-beta with TT. If no moves, return -1.
    moves = valid_moves(board)
    if not moves:
        return -1
    best = find_best_move(board, player, model, max_depth=depth, time_ms=time_ms)
    if best in moves:
        return best
    # Fallback: greedy evaluation
    return choose_action(board, player, model, epsilon=0.0)


def human_vs_ai(model: Model, depth: int = DEFAULT_SEARCH_DEPTH, time_ms: Optional[int] = None) -> None:
    print("Welcome to Connect 4")
    print("You are X, AI is O")
    while True:
        first = input("Do you want to move first? y or n: ").strip().lower()
        if first in ("y", "n"):
            break
    human_player = 1 if first == "y" else -1
    board = empty_board()
    current = 1  # player one starts
    print_board(board)
    while True:
        if current == human_player:
            # human turn
            while True:
                try:
                    move = input("Your move, choose a column 0 to 6 (or q to quit): ").strip().lower()
                    if move == 'q':
                        print("Bye")
                        return
                    if move in ("1","2","3","4","5","6","7"):
                        col = int(move) - 1
                    else:
                        col = int(move)
                    if col not in valid_moves(board):
                        print("That column is full or out of range. Try again.")
                        continue
                    break
                except ValueError:
                    print("Please enter a number 0 to 6 or q to quit")
            drop_piece(board, col, current)
            print_board(board)
            if check_winner(board, current):
                print("You win! Well played.")
                return
            if is_full(board):
                print("Draw game.")
                return
            current *= -1
        else:
            # AI turn
            col = ai_move(board, current, model, depth=depth, time_ms=time_ms)
            drop_piece(board, col, current)
            print(f"AI selects column {col}")
            print_board(board)
            if check_winner(board, current):
                print("AI wins.")
                return
            if is_full(board):
                print("Draw game.")
                return
            current *= -1


def evaluate(model: Model, episodes: int = 500, seed: Optional[int] = None, depth: int = DEFAULT_SEARCH_DEPTH, time_ms: Optional[int] = None) -> None:
    random.seed(seed)
    def run_vs_random(starts_as: int) -> Tuple[int, int, int]:
        w = d = l = 0
        for _ in range(episodes):
            board = empty_board()
            current = 1
            while True:
                if current == starts_as:
                    # AI move
                    col = ai_move(board, current, model, depth=depth, time_ms=time_ms)
                else:
                    # random baseline
                    col = random.choice(valid_moves(board)) if valid_moves(board) else -1
                drop_piece(board, col, current)
                if check_winner(board, current):
                    winner = current
                    break
                if is_full(board):
                    winner = 0
                    break
                current *= -1
            if winner == starts_as:
                w += 1
            elif winner == 0:
                d += 1
            else:
                l += 1
        return w, d, l

    w1, d1, l1 = run_vs_random(starts_as=1)
    w2, d2, l2 = run_vs_random(starts_as=-1)
    total = episodes * 2
    w = w1 + w2
    d = d1 + d2
    l = l1 + l2
    print(f"Evaluation vs random over {total} games")
    print(f"Wins {w}  Draws {d}  Losses {l}")
    print(f"Win rate {(w/total)*100:.1f}%  Draw rate {(d/total)*100:.1f}%  Loss rate {(l/total)*100:.1f}%")


def parse_args():
    p = argparse.ArgumentParser(description="Connect 4 self play learner and human vs AI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train a model by self play")
    p_train.add_argument("--episodes", type=int, default=20000, help="number of self play games")
    p_train.add_argument("--alpha", type=float, default=0.01, help="learning rate")
    p_train.add_argument("--gamma", type=float, default=0.98, help="discount for later moves")
    p_train.add_argument("--eps-start", type=float, default=1.0, help="initial exploration rate")
    p_train.add_argument("--eps-end", type=float, default=0.05, help="minimum exploration rate")
    p_train.add_argument("--eps-decay", type=float, default=0.999, help="multiplicative exploration decay per episode")
    p_train.add_argument("--seed", type=int, default=None, help="random seed")
    # Optional: use search policy during training
    p_train.add_argument("--train-depth", type=int, default=None, help="use alpha-beta search of this depth for training policy (with epsilon exploration)")
    p_train.add_argument("--train-time-ms", type=int, default=None, help="use time-limited alpha-beta for training policy (overrides depth if set)")
    p_train.add_argument("--model", type=str, default="model.json", help="path to save model")

    p_eval = sub.add_parser("eval", help="evaluate a model vs random player")
    p_eval.add_argument("--episodes", type=int, default=500, help="games per starting side")
    p_eval.add_argument("--model", type=str, default="model.json", help="model path")
    p_eval.add_argument("--seed", type=int, default=None, help="random seed")
    p_eval.add_argument("--depth", type=int, default=DEFAULT_SEARCH_DEPTH, help="search depth for AI moves")
    p_eval.add_argument("--time-ms", type=int, default=None, help="time budget per move in milliseconds (overrides deeper search if set)")

    p_play = sub.add_parser("play", help="play human vs AI")
    p_play.add_argument("--model", type=str, default="model.json", help="model path")
    p_play.add_argument("--depth", type=int, default=DEFAULT_SEARCH_DEPTH, help="search depth for AI moves")
    p_play.add_argument("--time-ms", type=int, default=None, help="time budget per AI move in milliseconds")

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "train":
        try:
            model = Model.load(args.model)
            print(f"Loaded existing model from {args.model}")
        except Exception:
            model = Model.new()
            print("Started new model")
        cfg = TrainConfig(
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
            seed=args.seed,
            train_depth=args.train_depth,
            train_time_ms=args.train_time_ms,
        )
        train(model, cfg, save_path=args.model)
    elif args.cmd == "eval":
        model = Model.load(args.model)
        evaluate(model, episodes=args.episodes, seed=args.seed, depth=args.depth, time_ms=args.time_ms)
    elif args.cmd == "play":
        model = Model.load(args.model)
        human_vs_ai(model, depth=args.depth, time_ms=args.time_ms)
    else:
        raise ValueError("unknown command")


if __name__ == "__main__":
    main()
