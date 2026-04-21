import io
import os
from functools import lru_cache
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_CANDIDATES = [
    BASE_DIR / "weights" / "flattened_centipawn_summary.keras",
    BASE_DIR / "weights" / "flattened_centi_pawn_summary.keras",
]
DEFAULT_STOCKFISH_CANDIDATES = [
    "/opt/homebrew/bin/stockfish",
    "/usr/local/bin/stockfish",
    "stockfish",
]
STOCKFISH_DEPTH = int(os.environ.get("STOCKFISH_DEPTH", "12"))
CENTIPAWN_CLAMP = 1000
REQUIRED_MOVES_PER_SIDE = 20

app = Flask(__name__, static_folder="static")


def score_to_centipawn(score: chess.engine.PovScore) -> int:
    relative_score = score.relative
    if relative_score.is_mate():
        mate_score = relative_score.mate()
        return 100000 if mate_score and mate_score > 0 else -100000
    return relative_score.score()


def clamp_centipawns(values: list[int]) -> list[int]:
    return [max(-CENTIPAWN_CLAMP, min(CENTIPAWN_CLAMP, value)) for value in values]


def resolve_model_path() -> Path:
    configured_path = os.environ.get("ELO_MODEL_PATH")
    if configured_path:
        return Path(configured_path)

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    return DEFAULT_MODEL_CANDIDATES[0]


def resolve_stockfish_path() -> str:
    configured_path = os.environ.get("STOCKFISH_PATH")
    if configured_path:
        return configured_path

    for candidate in DEFAULT_STOCKFISH_CANDIDATES:
        if candidate == "stockfish" or Path(candidate).exists():
            return candidate

    return DEFAULT_STOCKFISH_CANDIDATES[0]


def parse_optional_elo(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "?":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_pgn_game(file_bytes: bytes) -> chess.pgn.Game:
    text_stream = io.StringIO(file_bytes.decode("utf-8", errors="replace"))
    game = chess.pgn.read_game(text_stream)
    if game is None:
        raise ValueError("The uploaded file does not contain a readable PGN game.")
    return game


def analyze_game_centipawns(moves: list[chess.Move]) -> list[int]:
    board = chess.Board()
    centipawns: list[int] = []
    limit = chess.engine.Limit(depth=STOCKFISH_DEPTH)
    stockfish_path = resolve_stockfish_path()

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        engine.configure({"Threads": 2, "Hash": 256})
        for move in moves:
            info = engine.analyse(board, limit, info=chess.engine.INFO_SCORE)
            centipawns.append(score_to_centipawn(info["score"]))
            board.push(move)

    return clamp_centipawns(centipawns)


def build_feature_vector(centipawns: list[int]) -> np.ndarray:
    white_centipawn = centipawns[::2]
    black_centipawn = centipawns[1::2]

    if len(white_centipawn) < REQUIRED_MOVES_PER_SIDE or len(black_centipawn) < REQUIRED_MOVES_PER_SIDE:
        raise ValueError(
            "The PGN must contain at least 40 total plies so the model has 20 white and 20 black centipawn values."
        )

    white_first = white_centipawn[:REQUIRED_MOVES_PER_SIDE]
    black_first = black_centipawn[:REQUIRED_MOVES_PER_SIDE]

    summary_stats = [
        float(np.mean(white_centipawn)),
        float(np.std(white_centipawn)),
        float(np.median(white_centipawn)),
        float(np.max(white_centipawn)),
        float(np.min(white_centipawn)),
        float(np.mean(black_centipawn)),
        float(np.std(black_centipawn)),
        float(np.median(black_centipawn)),
        float(np.max(black_centipawn)),
        float(np.min(black_centipawn)),
    ]

    feature_vector = np.array(white_first + black_first + summary_stats, dtype=np.float32)
    return feature_vector.reshape(1, -1)


@lru_cache(maxsize=1)
def load_model() -> keras.Model:
    model_path = resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run the notebook cell that saves the model first."
        )
    return keras.models.load_model(model_path)


def predict_elo(file_bytes: bytes) -> dict:
    game = parse_pgn_game(file_bytes)
    moves = list(game.mainline_moves())
    centipawns = analyze_game_centipawns(moves)
    features = build_feature_vector(centipawns)
    model = load_model()
    prediction = float(model.predict(features, verbose=0)[0][0])
    white_elo = parse_optional_elo(game.headers.get("WhiteElo"))
    black_elo = parse_optional_elo(game.headers.get("BlackElo"))
    actual_game_elo = None
    if white_elo is not None and black_elo is not None:
        actual_game_elo = round((white_elo + black_elo) / 2)

    return {
        "prediction": round(prediction),
        "total_plies": len(centipawns),
        "white_plies": len(centipawns[::2]),
        "black_plies": len(centipawns[1::2]),
        "white_elo": white_elo,
        "black_elo": black_elo,
        "actual_game_elo": actual_game_elo,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("pgn_file")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please choose a PGN file."
        elif not uploaded_file.filename.lower().endswith(".pgn"):
            error = "Please upload a file with the .pgn extension."
        else:
            try:
                prediction = predict_elo(uploaded_file.read())
            except Exception as exc:
                error = str(exc)

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)
