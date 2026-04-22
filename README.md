# Chess Gameplay Strategy and ELO Analysis

Brief description: A data science project that analyzes Lichess game data to study gameplay patterns, relate strategy features to outcomes and player strength, and test whether player ELO can be predicted from move sequences and engine evaluations.

## Table of Contents

- [Project Overview](#project-overview)
- [Research Questions](#research-questions)
- [Dataset](#dataset)
- [Methods](#methods)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Interactive ELO Guesser](#interactive-elo-guesser)
- [Limitations and Future Work](#limitations-and-future-work)

## Project Overview

This repository contains the final submission for a chess-focused data science project built around online games from Lichess. The project explores whether large-scale game data can reveal meaningful differences in player behavior, identify useful correlations between strategy and outcomes, and support machine learning models that estimate player rating from moves alone.

The core workflow in the final notebook combines exploratory analysis, feature engineering from raw move lists, Stockfish-based centipawn evaluation, regression models, and neural network experiments.

Primary notebook: [final-submission.ipynb](/Users/thomasgrinstead/Developer/chess-ds-final/final-submission.ipynb)

## Research Questions

The final submission centers on three questions:

1. Can patterns in the data differentiate the gameplay style of experienced players from inexperienced players?
2. What correlations or other predictions can be identified from chess game data?
3. Can machine learning predict player rating based purely on a list of moves?

## Dataset

- Source: [Kaggle Chess Games Dataset](https://www.kaggle.com/datasets/datasnaek/chess/data)
- Platform: Lichess
- Format: CSV
- Includes game metadata such as winner, number of turns, rating, opening information, and move lists in standard chess notation

During cleaning and preprocessing, the project:

- removed player and game identifiers
- dropped non-rated games for rating-focused analysis
- encoded categorical outcomes numerically
- split move strings into structured arrays
- added average game ELO as a derived feature

## Methods

The analysis in the final notebook combines several stages:

- Exploratory analysis:
  histograms, correlation matrices, opening trends, game length analysis, and rating distributions
- Board-state feature engineering:
  move-by-move simulation with `python-chess` to extract checks, captures, forks, pins, attacks, defenses, center control, promotions, and piece movement counts
- Stockfish analysis:
  centipawn evaluations computed across games, repeated across multiple runs and averaged for final use
- Statistical modeling:
  linear, ridge, and lasso regression used to test relationships between strategy features, centipawn features, winners, and rating
- Machine learning:
  multiple regression-style neural network experiments using board-strategy features, centipawn sequences, and centipawn summary statistics

## Key Findings

- Basic board-strategy features showed weak correlation with player rating.
- Center control remained important across all skill levels and was slightly associated with winning.
- Higher-rated players tended to have lower average centipawn loss, meaning their moves were generally stronger by engine evaluation.
- ELO was a reasonably strong predictor of winner on its own, with about 61% accuracy in the notebook's simple predictor.
- Regression and neural-network models were not able to predict player rating accurately enough to support the original hypothesis.
- The strongest model inputs were raw centipawn sequences plus centipawn summary statistics, but performance was still not strong enough for reliable ELO prediction.

## Project Structure

- [final-submission.ipynb](/Users/thomasgrinstead/Developer/chess-ds-final/final-submission.ipynb): final report, analysis, models, and conclusions
- [data/games.csv](/Users/thomasgrinstead/Developer/chess-ds-final/data/games.csv): chess game dataset used in the project
- [accuracy.py](/Users/thomasgrinstead/Developer/chess-ds-final/accuracy.py): supporting script for model evaluation
- [interactive/app.py](/Users/thomasgrinstead/Developer/chess-ds-final/interactive/app.py): Flask app for the interactive ELO guesser
- [weights/flattened_centipawn_summary.keras](/Users/thomasgrinstead/Developer/chess-ds-final/weights/flattened_centipawn_summary.keras): saved model weights for the demo application
- [centipawn/](/Users/thomasgrinstead/Developer/chess-ds-final/centipawn): exported centipawn evaluation runs

## Setup

The notebook notes the following environment setup:

```bash
conda create -n chess python=3.11
conda activate chess
pip install pandas
pip install matplotlib
pip install seaborn
pip install chess
pip install tensorflow
pip install flask
pip install pathos
```

Stockfish is also required for the engine-based analysis and the interactive demo:

- [Stockfish Download](https://stockfishchess.org/download/)

## Interactive ELO Guesser

The project includes a small Flask app that accepts a PGN file and estimates the average ELO of the players using the best-performing model from the notebook.

To run it:

```bash
export FLASK_APP=interactive/app.py
flask run
```

Then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

Demo screenshot:

![Interactive ELO Guesser](/Users/thomasgrinstead/Developer/chess-ds-final/gte.png)

## Limitations and Future Work

The final notebook concludes that chess is difficult to reduce to a small set of interpretable numerical features. While the project successfully built a substantial preprocessing and analysis pipeline, the extracted features were not sufficient for strong rating prediction.

Future improvements suggested by the submission include:

- using a broader and more diverse dataset
- developing richer strategy features
- expanding engine-based analysis
- exploring deeper modeling approaches beyond simple regression pipelines
