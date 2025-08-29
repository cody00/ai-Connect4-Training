#!/usr/bin/env python3
"""
Simple clickable GUI for Human vs AI Connect 4 using Tkinter.

Usage:
  python connect4_gui.py --model model.json

Click a column to drop your piece. Use the controls to choose who starts
and to reset the game.
"""
from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import messagebox
from typing import Optional

# Reuse core game + model from the CLI implementation
from connect4_ai import (
    ROWS,
    COLS,
    empty_board,
    drop_piece,
    valid_moves,
    check_winner,
    is_full,
    ai_move,
    Model,
    DEFAULT_SEARCH_DEPTH,
)


CELL = 90  # pixel size of one cell
PADDING = 20  # outer padding
DISC_PAD = 8  # disc inner padding to show board background


class Connect4GUI:
    def __init__(self, root: tk.Tk, model: Model, human_first: bool = True, depth: int = DEFAULT_SEARCH_DEPTH, time_ms: Optional[int] = None) -> None:
        self.root = root
        self.model = model
        self.depth = depth
        self.time_ms = time_ms
        self.human_player = 1 if human_first else -1
        self.current = 1
        self.board = empty_board()
        self.game_over = False

        root.title("Connect 4 — Human vs AI")
        self.status_var = tk.StringVar()
        self.status_var.set("You are Red (X). Click a column to move.")

        # Layout frames
        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X)

        self.start_var = tk.StringVar(value="human")
        tk.Label(top, text="First move:").pack(side=tk.LEFT, padx=(8, 4))
        tk.Radiobutton(top, text="Human", variable=self.start_var, value="human").pack(side=tk.LEFT)
        tk.Radiobutton(top, text="AI", variable=self.start_var, value="ai").pack(side=tk.LEFT)
        tk.Button(top, text="New Game", command=self.new_game).pack(side=tk.LEFT, padx=12)

        tk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        width = COLS * CELL + 2 * PADDING
        height = ROWS * CELL + 2 * PADDING
        self.canvas = tk.Canvas(root, width=width, height=height, bg="#1E3A8A", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self.on_click)

        self.draw_board()

        # If AI starts, let it move after initial render
        if self.human_player != self.current:
            self.root.after(300, self.ai_turn)

    def new_game(self) -> None:
        self.board = empty_board()
        self.current = 1
        self.game_over = False
        self.human_player = 1 if self.start_var.get() == "human" else -1
        self.status_var.set("You are Red (X). Click a column to move." if self.human_player == 1 else "AI starts. Please wait…")
        self.draw_board()
        if self.human_player != self.current:
            self.root.after(300, self.ai_turn)

    def draw_board(self) -> None:
        self.canvas.delete("all")
        # Draw board background
        width = COLS * CELL
        height = ROWS * CELL
        self.canvas.create_rectangle(PADDING, PADDING, PADDING + width, PADDING + height, fill="#1E3A8A", outline="")

        # Draw the circular slots with current discs
        for r in range(ROWS):
            for c in range(COLS):
                x0 = PADDING + c * CELL + DISC_PAD
                y0 = PADDING + r * CELL + DISC_PAD
                x1 = PADDING + (c + 1) * CELL - DISC_PAD
                y1 = PADDING + (r + 1) * CELL - DISC_PAD
                val = self.board[r][c]
                color = "#0F172A"  # empty slot (near-black/blue)
                if val == 1:
                    color = "#DC2626"  # red for human (X)
                elif val == -1:
                    color = "#FACC15"  # yellow for AI (O)
                self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline="#0B1220")

        # Draw column hover guidance lines (subtle)
        for c in range(COLS + 1):
            x = PADDING + c * CELL
            self.canvas.create_line(x, PADDING, x, PADDING + height, fill="#0B1220")

    def column_from_x(self, x: int) -> Optional[int]:
        x_rel = x - PADDING
        if x_rel < 0:
            return None
        c = x_rel // CELL
        if 0 <= c < COLS:
            return int(c)
        return None

    def on_click(self, event) -> None:
        if self.game_over:
            return
        if self.current != self.human_player:
            return  # wait for AI

        col = self.column_from_x(event.x)
        if col is None:
            return
        if col not in valid_moves(self.board):
            self.status_var.set("That column is full. Try another.")
            return

        # Human move
        drop_piece(self.board, col, self.current)
        self.draw_board()
        if self.check_end_after_move():
            return
        self.current *= -1
        self.status_var.set("AI is thinking…")
        self.root.after(250, self.ai_turn)

    def ai_turn(self) -> None:
        if self.game_over:
            return
        if self.current == self.human_player:
            return
        # AI selects a column
        col = ai_move(self.board, self.current, self.model, depth=self.depth, time_ms=self.time_ms)
        if col not in valid_moves(self.board):
            # Fallback: if something odd happens pick any valid move
            moves = valid_moves(self.board)
            if not moves:
                self.declare_draw()
                return
            col = moves[0]
        drop_piece(self.board, col, self.current)
        self.draw_board()
        if self.check_end_after_move(ai=True, col=col):
            return
        self.current *= -1
        self.status_var.set("Your turn. Click a column.")

    def check_end_after_move(self, ai: bool = False, col: Optional[int] = None) -> bool:
        if check_winner(self.board, self.current):
            if self.current == self.human_player:
                self.declare_win()
            else:
                self.declare_loss(col)
            return True
        if is_full(self.board):
            self.declare_draw()
            return True
        return False

    def declare_win(self) -> None:
        self.game_over = True
        self.status_var.set("You win! 🎉 Click New Game to play again.")
        messagebox.showinfo("Game Over", "You win! Well played.")

    def declare_loss(self, col: Optional[int]) -> None:
        self.game_over = True
        msg = f"AI wins. (col {col})" if col is not None else "AI wins."
        self.status_var.set(msg + " Click New Game to play again.")
        messagebox.showinfo("Game Over", "AI wins.")

    def declare_draw(self) -> None:
        self.game_over = True
        self.status_var.set("Draw game. Click New Game to play again.")
        messagebox.showinfo("Game Over", "Draw game.")


def parse_args():
    p = argparse.ArgumentParser(description="Tkinter GUI to play Connect 4 vs AI")
    p.add_argument("--model", type=str, default="model.json", help="model path (json)")
    p.add_argument("--ai-first", action="store_true", help="AI makes the first move")
    p.add_argument("--depth", type=int, default=DEFAULT_SEARCH_DEPTH, help="search depth for AI moves")
    p.add_argument("--time-ms", type=int, default=None, help="time budget per AI move in milliseconds")
    return p.parse_args()


def main():
    args = parse_args()
    model = Model.load(args.model)
    root = tk.Tk()
    gui = Connect4GUI(root, model, human_first=not args.ai_first, depth=args.depth, time_ms=args.time_ms)
    root.mainloop()


if __name__ == "__main__":
    main()
