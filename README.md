# Cinema AI: Box Office Success Predictor

### Overview
A Machine Learning web application that predicts movie box office revenue based on the cast, crew, budget, and genre. It features a "Battle Mode" to compare two casting scenarios side-by-side to determine the highest ROI.

### Features
* **Revenue Prediction:** Uses Random Forest Regressor to estimate box office earnings.
* **Star Power Engine:** Custom algorithm that quantifies the market value of Actors, Directors, and Composers.
* **Battle Mode:** A Business Intelligence tool to compare "Strategy A vs. Strategy B".
* **Dynamic Visuals:** Fetches real-time cast images via Wikipedia API and renders interactive Plotly charts.

### Tech Stack
* **Python** (Pandas, NumPy, Scikit-Learn)
* **Frontend:** Streamlit
* **Visualization:** Plotly, Seaborn, Matplotlib
* **Data Sources:** TMDB Dataset + Custom Indian Cinema Data Injection

### How to Run
1.  Install requirements: `pip install -r requirements.txt`
2.  Run the app: `streamlit run app.py`