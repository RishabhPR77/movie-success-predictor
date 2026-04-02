# 🎬 CinemaIQ — Box Office Revenue Predictor

ML-powered web app that predicts box office revenue for South Asian films
using budget, cast, director, genre, and release timing.

## Live Demo
[[cinemaiq.streamlit.app](https://movie-success-predictor-xu2vnp53g3jg8a3pmrr3k3.streamlit.app/)](your-link-here)

## Results
- Best Model: XGBoost (or whatever yours is)
- R² Score: 0.XX on held-out test set
- MAE: $XX.XM

## Tech Stack
Python · Scikit-learn · XGBoost · Streamlit · Plotly · Groq AI (LLaMA 3.3)

## Features
- Predicts revenue across pessimistic / base / optimistic scenarios
- Head-to-head strategy battle between two production configurations
- Market insights: optimal release month, genre analysis, talent power index
- AI analyst commentary powered by Groq LLaMA 3.3 70B

## Pipeline
cleaningdata.ipynb → preprocess.ipynb → train.ipynb → app.py
