"""
Calculate chess game accuracy scores from move lists.

The public API stays centered on `get_accuracies(...)`, which returns one row
per game with white/black/overall accuracy plus average centipawn loss.
"""

from __future__ import annotations

import math
from contextlib import closing
from typing import Iterable

import chess
import chess.engine
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
SF_DEPTH = 15
MATE_CP = 1_000


# ── Internal helpers ──────────────────────────────────────────────────────────

def _accuracy_from_cpl(avg_cpl: float) -> float:
    acc = 103.1668 * math.exp(-0.04354 * avg_cpl) - 3.1669
    return round(max(0.0, min(100.0, acc)), 2)


def _position_cache_key(board: chess.Board) -> str:
    return board.epd()


def _score_to_white_cp(score: chess.engine.PovScore) -> int:
    return score.white().score(mate_score=MATE_CP)


def _get_position_eval_cp(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    cache: dict[str, int],
) -> int:
    key = _position_cache_key(board)
    cached = cache.get(key)
    if cached is not None:
        return cached

    info = engine.analyse(board, limit, info=chess.engine.INFO_SCORE)
    cp = _score_to_white_cp(info["score"])
    cache[key] = cp
    return cp


def _parse_move(board: chess.Board, move_text: str) -> chess.Move:
    move_text = move_text.strip()
    if not move_text:
        raise ValueError("Empty move string")

    try:
        move = chess.Move.from_uci(move_text)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    return board.parse_san(move_text)


def _normalize_moves(moves: object) -> list[str]:
    if moves is None:
        return []
    if isinstance(moves, str):
        return [token for token in moves.split() if token]
    if isinstance(moves, Iterable):
        return [str(token).strip() for token in moves if str(token).strip()]
    return []


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _analyze_one(
    moves: object,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    cache: dict[str, int],
) -> dict:
    board = chess.Board()
    white_losses: list[int] = []
    black_losses: list[int] = []
    normalized_moves = _normalize_moves(moves)

    if not normalized_moves:
        return _EMPTY_ROW.copy()

    current_eval = _get_position_eval_cp(board, engine, limit, cache)

    for ply_index, move_text in enumerate(normalized_moves):
        try:
            move = _parse_move(board, move_text)
        except ValueError:
            continue

        board.push(move)
        next_eval = _get_position_eval_cp(board, engine, limit, cache)

        if ply_index % 2 == 0:
            white_losses.append(max(0, current_eval - next_eval))
        else:
            black_losses.append(max(0, next_eval - current_eval))

        current_eval = next_eval

        if board.is_game_over():
            break

    if not white_losses and not black_losses:
        return _EMPTY_ROW.copy()

    white_avg_cpl = _mean(white_losses)
    black_avg_cpl = _mean(black_losses)
    overall_avg_cpl = _mean(white_losses + black_losses)

    return {
        "white_accuracy": _accuracy_from_cpl(white_avg_cpl),
        "black_accuracy": _accuracy_from_cpl(black_avg_cpl),
        "overall_accuracy": _accuracy_from_cpl(overall_avg_cpl),
        "white_avg_cpl": round(white_avg_cpl, 1),
        "black_avg_cpl": round(black_avg_cpl, 1),
    }


_EMPTY_ROW = {
    "white_accuracy": None,
    "black_accuracy": None,
    "overall_accuracy": None,
    "white_avg_cpl": None,
    "black_avg_cpl": None,
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_accuracies(
    games: list,
    depth: int = SF_DEPTH,
    stockfish_path: str = STOCKFISH_PATH,
) -> pd.DataFrame:
    """
    Analyze a list of games and return a DataFrame of accuracy scores.

    Parameters
    ----------
    games : list[list[str] | str | None]
        Each element is either a list/tuple of move strings or a single
        whitespace-delimited move string. Both UCI and SAN moves are accepted.
    depth : int
        Stockfish search depth per position.
    stockfish_path : str
        Path to the Stockfish binary.
    """
    limit = chess.engine.Limit(depth=depth)
    cache: dict[str, int] = {}
    rows = []

    with closing(chess.engine.SimpleEngine.popen_uci(stockfish_path)) as engine:
        for moves in games:
            rows.append(_analyze_one(moves, engine, limit, cache))

    return pd.DataFrame(rows)
