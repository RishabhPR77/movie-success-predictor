import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Cinema AI: Box Office Predictor", page_icon="🎬", layout="wide")

# --- 2. LOAD SAVED MODELS ---
@st.cache_resource
def load_artifacts():
    try:
        with open('movie_revenue_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return model, artifacts
    except FileNotFoundError:
        return None, None

model, artifacts = load_artifacts()

if model is None:
    st.error("⚠️ System Error: Models not found. Please run 'train_model.py' first.")
    st.stop()

# --- 3. INTELLIGENT FUNCTIONS ---
def get_combined_power_score(names, type_dict):
    """Calculates the average market value of the selected cast/crew."""
    if not names: return 0
    if isinstance(names, str): names = [names]
    scores = [type_dict.get(n, 0) for n in names]
    return np.mean(scores) if scores else 0

@st.cache_data
def fetch_image(name):
    """Fetches image from Wikipedia using Fuzzy Search (Tolerates spelling errors)."""
    try:
        headers = {"User-Agent": "MovieApp/1.0"}
        search_url = "https://en.wikipedia.org/w/api.php"
        
        # 1. Fuzzy Search to find the page title
        search_params = {
            "action": "query", "list": "search", "srsearch": name, "format": "json", "srlimit": 1
        }
        res = requests.get(search_url, params=search_params, headers=headers, timeout=2).json()
        if not res.get("query", {}).get("search"): return None
        
        # 2. Get Image for that specific page
        title = res["query"]["search"][0]["title"]
        img_params = {
            "action": "query", "titles": title, "prop": "pageimages", "format": "json", "pithumbsize": 200
        }
        img_res = requests.get(search_url, params=img_params, headers=headers, timeout=2).json()
        pages = img_res["query"]["pages"]
        for p in pages:
            if "thumbnail" in pages[p]: return pages[p]["thumbnail"]["source"]
    except:
        return None
    return None

def predict_revenue_core(budget, runtime, vote_avg, vote_count, month, actors, director, composer, genres):
    """Core logic to predict revenue based on inputs."""
    # 1. Calculate Power Scores
    s_score = get_combined_power_score(actors, artifacts['actor_power'])
    d_score = get_combined_power_score(director, artifacts['director_power'])
    c_score = get_combined_power_score(composer, artifacts['composer_power'])
    
    # 2. Process Genres
    g_input = {f"Genre_{g}": 0 for g in artifacts['mlb'].classes_}
    for g in genres: g_input[f"Genre_{g}"] = 1
    
    # 3. Create Dataframe
    input_data = pd.DataFrame({
        'budget': [budget], 'runtime': [runtime], 'vote_average': [vote_avg],
        'vote_count': [vote_count], 'release_month': [month],
        'star_power': [s_score], 'director_power': [d_score], 'composer_power': [c_score],
        **g_input
    })
    
    return model.predict(input_data)[0]

# --- 4. APP LAYOUT ---
st.title("🎬 Cinema AI: Blockbuster Analytics")
st.markdown("### Use Artificial Intelligence to predict box office success and compare scenarios.")

# TABS: The Professional Touch
tab1, tab2 = st.tabs(["🔮 Single Prediction", "⚔️ Battle Mode (Compare)"])

# ===========================
# TAB 1: SINGLE PREDICTOR
# ===========================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Financials & Hype")
        budget = st.number_input("Budget ($)", 1000, value=50000000, step=1000000, help="Total production cost")
        runtime = st.slider("Runtime (mins)", 60, 240, 150)
        vote_avg = st.slider("Target Rating (IMDB)", 1.0, 10.0, 7.5)
        month = st.selectbox("Release Month", range(1, 13), index=5, format_func=lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))

    with col2:
        st.subheader("🌟 Cast & Crew")
        actor_list = sorted(artifacts['actor_power'].keys())
        dir_list = sorted(artifacts['director_power'].keys())
        comp_list = sorted(artifacts['composer_power'].keys())
        
        actors = st.multiselect("Star Cast", actor_list, default=["Shah Rukh Khan"])
        
        # Show Faces (Visual Polish)
        if actors:
            cols = st.columns(len(actors))
            for idx, act in enumerate(actors):
                with cols[idx]:
                    img = fetch_image(act)
                    if img: st.image(img, width=80)
                    st.caption(act)

        director = st.selectbox("Director", dir_list, index=dir_list.index("S.S. Rajamouli") if "S.S. Rajamouli" in dir_list else 0)
        composer = st.selectbox("Composer", comp_list, index=comp_list.index("Anirudh Ravichander") if "Anirudh Ravichander" in comp_list else 0)
        genres = st.multiselect("Genres", artifacts['mlb'].classes_, default=["Action"])

    if st.button("🚀 Predict Box Office", type="primary"):
        prediction = predict_revenue_core(budget, runtime, vote_avg, 5000, month, actors, director, composer, genres)
        
        st.markdown("---")
        
        # Result Metrics
        c1, c2 = st.columns([1, 2])
        
        with c1:
            profit = prediction - budget
            roi = (profit / budget) * 100
            st.metric("Predicted Revenue", f"${prediction:,.0f}", delta=f"{roi:.0f}% ROI")
            
            if profit > 0:
                st.balloons()
                st.success(f"✅ **Hit!** Profit: ${profit:,.0f}")
            else:
                st.error(f"⚠️ **Flop.** Loss: ${abs(profit):,.0f}")
        
        with c2:
            # UPGRADE: Interactive Plotly Chart
            st.subheader("📊 Sensitivity Analysis")
            st.caption("Hover over the chart to see how Budget impacts Revenue for THIS cast.")
            
            # Generate 20 scenarios
            test_budgets = np.linspace(budget * 0.5, budget * 3, 20)
            test_revenues = [predict_revenue_core(b, runtime, vote_avg, 5000, month, actors, director, composer, genres) for b in test_budgets]
            
            # Plotly Line Chart
            fig = px.line(x=test_budgets, y=test_revenues, labels={'x': 'Budget ($)', 'y': 'Predicted Revenue ($)'})
            fig.add_vline(x=budget, line_dash="dash", line_color="red", annotation_text="Current Budget")
            fig.update_layout(hovermode="x unified") # Nice hover effect
            st.plotly_chart(fig, use_container_width=True)

# ===========================
# TAB 2: BATTLE MODE ⚔️
# ===========================
with tab2:
    st.header("⚔️ Movie Battle: Strategy A vs Strategy B")
    st.markdown("Compare two different casting/budgeting decisions side-by-side to find the most profitable option.")
    
    # Side-by-Side Layout
    col_a, col_b = st.columns(2)
    
    # --- MOVIE A INPUTS ---
    with col_a:
        st.subheader("🥊 Challenger A")
        st.info("e.g., High Budget + Global Star")
        a_budget = st.number_input("Budget A ($)", value=40000000, key="a_bud")
        a_actors = st.multiselect("Cast A", actor_list, default=["Allu Arjun"], key="a_act")
        a_dir = st.selectbox("Director A", dir_list, index=dir_list.index("Sukumar") if "Sukumar" in dir_list else 0, key="a_dir")
        
    # --- MOVIE B INPUTS ---
    with col_b:
        st.subheader("🛡️ Defender B")
        st.info("e.g., Medium Budget + Cult Classic Director")
        b_budget = st.number_input("Budget B ($)", value=30000000, key="b_bud")
        b_actors = st.multiselect("Cast B", actor_list, default=["Fahadh Faasil"], key="b_act")
        b_dir = st.selectbox("Director B", dir_list, index=dir_list.index("Lokesh Kanagaraj") if "Lokesh Kanagaraj" in dir_list else 0, key="b_dir")

    st.markdown("---")
    
    if st.button("⚔️ Fight! (Compare ROI)", type="primary"):
        # Run both predictions
        # (Using defaults for non-critical inputs to keep Battle Mode simple)
        rev_a = predict_revenue_core(a_budget, 150, 7.5, 5000, 6, a_actors, a_dir, "Anirudh Ravichander", ["Action"])
        rev_b = predict_revenue_core(b_budget, 150, 7.5, 5000, 6, b_actors, b_dir, "Anirudh Ravichander", ["Action"])
        
        profit_a = rev_a - a_budget
        profit_b = rev_b - b_budget
        
        # Display Results
        c1, c2, c3 = st.columns(3)
        
        c1.metric("Movie A Revenue", f"${rev_a:,.0f}", delta=f"${profit_a:,.0f} Profit")
        c2.metric("Movie B Revenue", f"${rev_b:,.0f}", delta=f"${profit_b:,.0f} Profit")
        
        diff = profit_a - profit_b
        if diff > 0:
            c3.success(f"🏆 **Strategy A Wins!**\n\nGenerates **${diff:,.0f}** more profit.")
        else:
            c3.error(f"🏆 **Strategy B Wins!**\n\nGenerates **${abs(diff):,.0f}** more profit.")
            
        # UPGRADE: Interactive Bar Chart
        fig_battle = go.Figure(data=[
            go.Bar(name='Movie A', x=['Revenue'], y=[rev_a], marker_color='#00CC96'),
            go.Bar(name='Movie B', x=['Revenue'], y=[rev_b], marker_color='#EF553B')
        ])
        fig_battle.update_layout(title="Revenue Face-off", barmode='group')
        st.plotly_chart(fig_battle, use_container_width=True)