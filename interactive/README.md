# Interactive Flask App

This folder contains a minimal Flask app for uploading a `.pgn` file and predicting ELO from the saved centipawn-summary neural network.

## Run

From the project root:

```bash
export FLASK_APP=interactive/app.py
flask run
```

## Requirements

- A saved model at `weights/flattened_centipawn_summary.keras`
- Stockfish installed locally
- Python packages: `flask`, `tensorflow`, `python-chess`, `numpy`

## Optional environment variables

- `STOCKFISH_PATH` to point at your Stockfish binary
- `STOCKFISH_DEPTH` to override the default analysis depth of `12`
- `ELO_MODEL_PATH` to point at a different saved Keras model
