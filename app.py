import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy
import os
from dotenv import load_dotenv

load_dotenv()

# ── TEMPORARY DEBUG (remove after fixing) ──
key = os.getenv("GROQ_API_KEY")
if key:
    st.sidebar.success(f"Groq key loaded: ...{key[-6:]}")
else:
    st.sidebar.error("GROQ_API_KEY not found in environment")


# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CinemaIQ | Box Office Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# 2. GROQ CLIENT
# ─────────────────────────────────────────────
try:
    from groq import Groq
    _groq_key = os.getenv("GROQ_API_KEY")
    groq_client = Groq(api_key=_groq_key) if _groq_key else None
except ImportError:
    groq_client = None

# ─────────────────────────────────────────────
# 3. GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]          { font-family: 'Inter', sans-serif; }
.stApp                              { background: linear-gradient(135deg, #07070f 0%, #0d0d1c 100%); }
#MainMenu, footer, header           { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #11112a 0%, #18183d 50%, #0e0e28 100%);
    border: 1px solid rgba(255,215,0,.18);
    border-radius: 24px; padding: 48px 40px; margin-bottom: 28px;
    text-align: center; position: relative; overflow: hidden;
}
.hero::after {
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse at 50% 0%, rgba(255,215,0,.07) 0%, transparent 65%);
    pointer-events:none;
}
.hero h1 {
    font-size: 3rem; font-weight: 800; margin: 0; letter-spacing: -1px;
    background: linear-gradient(90deg,#FFD700,#FFAA00,#FF6B35);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p   { color:rgba(255,255,255,.5); font-size:1.05rem; margin:10px 0 0; }
.badges   { margin-top:18px; display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }
.badge    {
    background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
    border-radius:20px; padding:5px 14px; font-size:.77rem; color:rgba(255,255,255,.55);
}
.glass-card {
    background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07);
    border-radius:18px; padding:24px;
}
.result-card {
    background:linear-gradient(135deg,#191932,#202044);
    border:1px solid rgba(255,215,0,.22); border-radius:20px; padding:32px;
    text-align:center;
}
.result-revenue { font-size:3rem; font-weight:800; color:#FFD700; line-height:1; }
.result-label   { font-size:.72rem; text-transform:uppercase; letter-spacing:2px;
                  color:rgba(255,255,255,.4); margin-bottom:10px; }
.kpi-card {
    background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
    border-radius:14px; padding:20px; text-align:center;
}
.kpi-value { font-size:1.6rem; font-weight:700; color:#fff; }
.kpi-label { font-size:.7rem; text-transform:uppercase; letter-spacing:1.5px;
             color:rgba(255,255,255,.4); margin-top:4px; }
.kpi-pos   { color:#00CC96; font-size:.82rem; margin-top:5px; }
.kpi-neg   { color:#EF553B; font-size:.82rem; margin-top:5px; }
.kpi-gold  { color:#FFD700; font-size:.82rem; margin-top:5px; }
.sec-title {
    font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:1.8px;
    color:rgba(255,255,255,.85); border-left:3px solid #FFD700;
    padding-left:10px; margin:20px 0 14px;
}
.divider {
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(255,215,0,.25),transparent);
    margin:24px 0;
}
.stTabs [data-baseweb="tab-list"] {
    background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07);
    border-radius:12px; padding:4px; gap:4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px; color:rgba(255,255,255,.5);
    font-weight:500; font-size:.9rem; padding:9px 22px;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#FFD700,#FFA500) !important;
    color:#000 !important; font-weight:700 !important;
}
.stButton > button {
    background:linear-gradient(135deg,#FFD700,#FFAA00) !important;
    color:#000 !important; font-weight:700 !important; border:none !important;
    border-radius:12px !important; padding:14px 32px !important;
    font-size:.95rem !important; width:100%; letter-spacing:.4px;
    box-shadow:0 4px 22px rgba(255,215,0,.2) !important; transition:all .2s !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 32px rgba(255,215,0,.35) !important;
}
.verdict {
    border-radius:14px; padding:20px; text-align:center; font-size:1.05rem; font-weight:600;
}
.verdict-win {
    background:rgba(0,204,150,.08); border:1px solid rgba(0,204,150,.3); color:#00CC96;
}
.power-tag {
    background:rgba(255,215,0,.15); border:1px solid rgba(255,215,0,.25);
    border-radius:8px; padding:2px 8px; font-size:.68rem; color:#FFD700; font-weight:600;
}
.placeholder {
    height:220px; display:flex; flex-direction:column; align-items:center;
    justify-content:center; background:rgba(255,255,255,.015);
    border:1px dashed rgba(255,255,255,.09); border-radius:20px; gap:12px;
}
.scenario-label-a { color:#00CC96; font-weight:700; font-size:.95rem; margin-bottom:14px; }
.scenario-label-b { color:#EF553B; font-weight:700; font-size:.95rem; margin-bottom:14px; }

/* ── AI Insight Card ── */
.insight-card {
    background: linear-gradient(135deg, rgba(99,102,241,.07), rgba(168,85,247,.05));
    border: 1px solid rgba(168,85,247,.28);
    border-radius: 14px; padding: 20px 24px; margin: 20px 0;
    position: relative;
}
.insight-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #6366f1, #a855f7, #6366f1);
    border-radius: 14px 14px 0 0;
}
.insight-label {
    font-size: .68rem; text-transform: uppercase; letter-spacing: 2px;
    color: rgba(168,85,247,.9); font-weight: 700; margin-bottom: 10px;
}
.insight-text {
    color: rgba(255,255,255,.72); font-size: .88rem; line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 4. PLOTLY BASE THEME + LAYOUT HELPER
# ─────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,.015)',
    font=dict(family='Inter', color='rgba(255,255,255,.65)', size=12),
    xaxis=dict(gridcolor='rgba(255,255,255,.05)', showline=False, zeroline=False),
    yaxis=dict(gridcolor='rgba(255,255,255,.05)', showline=False, zeroline=False),
    margin=dict(l=16, r=16, t=36, b=16),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
)

def make_layout(**overrides):
    layout = copy.deepcopy(LAYOUT)
    for key, val in overrides.items():
        if key in layout and isinstance(layout[key], dict) and isinstance(val, dict):
            layout[key].update(val)
        else:
            layout[key] = val
    return layout

MONTHS = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
          7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

# ─────────────────────────────────────────────
# 5. LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open('movie_revenue_model.pkl','rb') as f: model = pickle.load(f)
        with open('model_artifacts.pkl','rb') as f:    artifacts = pickle.load(f)
        return model, artifacts
    except FileNotFoundError:
        return None, None

model, artifacts = load_artifacts()

if model is None:
    st.markdown("""
    <div class='result-card' style='border-color:rgba(239,85,59,.4);max-width:480px;margin:120px auto;'>
        <div style='font-size:2rem;'>⚠️</div>
        <div style='color:#EF553B;font-size:1.3rem;font-weight:700;margin:14px 0 8px;'>Model Files Not Found</div>
        <div style='color:rgba(255,255,255,.45);'>Run <code>train_model.py</code> to generate the required artifacts.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# 6. CORE HELPERS
# ─────────────────────────────────────────────
def power_score(names, pdict):
    if not names: return 0.0
    if isinstance(names, str): names = [names]
    return float(np.mean([pdict.get(n, 0) for n in names]))

@st.cache_data(ttl=3600)
def fetch_image(name):
    try:
        h = {"User-Agent": "CinemaIQ/2.0"}
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action":"query","list":"search","srsearch":name,
                                 "format":"json","srlimit":1}, headers=h, timeout=3).json()
        if not r.get("query",{}).get("search"): return None
        title = r["query"]["search"][0]["title"]
        ir = requests.get("https://en.wikipedia.org/w/api.php",
                          params={"action":"query","titles":title,"prop":"pageimages",
                                  "format":"json","pithumbsize":300}, headers=h, timeout=3).json()
        pages = ir["query"]["pages"]
        for p in pages:
            if "thumbnail" in pages[p]: return pages[p]["thumbnail"]["source"]
    except:
        return None

def predict(budget, runtime, vote_avg, vote_count, month, actors, director, composer, genres):
    g = {f"Genre_{g}": 0 for g in artifacts['mlb'].classes_}
    for genre in genres: g[f"Genre_{genre}"] = 1

    row = pd.DataFrame({
        'log_budget'     : [np.log1p(budget)],        # ← was 'budget': [budget]
        'runtime'        : [runtime],
        'vote_average'   : [vote_avg],
        'vote_count'     : [vote_count],
        'release_month'  : [month],
        'star_power'     : [power_score(actors,   artifacts['actor_power'])],
        'director_power' : [power_score(director, artifacts['director_power'])],
        'composer_power' : [power_score(composer, artifacts['composer_power'])],
        **g
    })

    row = row[artifacts['feature_cols']]               # ← ensures correct column order
    log_pred = float(model.predict(row)[0])
    return float(np.expm1(log_pred))                   # ← converts log output back to dollars

def predict_scenarios(budget, runtime, vote_avg, vote_count, month, actors, director, composer, genres):
    base = predict(budget, runtime, vote_avg,            vote_count,         month, actors, director, composer, genres)
    pess = predict(budget, runtime, max(vote_avg-.8,1),  int(vote_count*.55), month, actors, director, composer, genres) * .78
    opt  = predict(budget, runtime, min(vote_avg+.7,10), int(vote_count*1.6), month, actors, director, composer, genres) * 1.22
    return pess, base, opt

# ─────────────────────────────────────────────
# 7. GROQ INSIGHT HELPERS
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a senior film industry analyst specializing in South Asian cinema box office performance. "
    "Give concise, specific, data-driven insights in 2-3 sentences. "
    "Reference the exact numbers provided. No bullet points, no headers, no markdown."
)

@st.cache_data(ttl=3600, show_spinner=False)
def get_groq_insight(prompt: str) -> str | None:
    if groq_client is None:
        return None   # key missing — silently skip
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=180,
            temperature=0.65
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Groq API error: {e}")  # now visible instead of silent
        return None

def show_insight(text: str | None):
    """Renders the styled AI insight card. Silently skips if text is None."""
    if not text:
        return
    st.markdown(f"""
    <div class='insight-card'>
        <div class='insight-label'>AI Analyst Insight</div>
        <div class='insight-text'>{text}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 8. CACHED INSIGHT DATA
# ─────────────────────────────────────────────
@st.cache_data
def get_month_revenues(_model, _artifacts):
    da = ["Shah Rukh Khan"] if "Shah Rukh Khan" in _artifacts['actor_power'] else [sorted(_artifacts['actor_power'])[0]]
    dd = "S.S. Rajamouli"   if "S. S. Rajamouli"   in _artifacts['director_power']  else sorted(_artifacts['director_power'])[0]
    dc = "Anirudh Ravichander" if "Anirudh Ravichander" in _artifacts['composer_power'] else sorted(_artifacts['composer_power'])[0]
    return [predict(50_000_000,150,7.5,5000,m,da,dd,dc,["Action"]) for m in range(1,13)]

@st.cache_data
def get_genre_revenues(_model, _artifacts):
    da = ["Shah Rukh Khan"] if "Shah Rukh Khan" in _artifacts['actor_power'] else [sorted(_artifacts['actor_power'])[0]]
    dd = "S.S. Rajamouli"   if "S. S. Rajamouli"   in _artifacts['director_power']  else sorted(_artifacts['director_power'])[0]
    dc = "Anirudh Ravichander" if "Anirudh Ravichander" in _artifacts['composer_power'] else sorted(_artifacts['composer_power'])[0]
    return {g: predict(50_000_000,150,7.5,5000,6,da,dd,dc,[g]) for g in _artifacts['mlb'].classes_}

# ─────────────────────────────────────────────
# 9. SORTED LISTS
# ─────────────────────────────────────────────
actor_list = sorted(artifacts['actor_power'].keys())
dir_list   = sorted(artifacts['director_power'].keys())
comp_list  = sorted(artifacts['composer_power'].keys())

# ─────────────────────────────────────────────
# 10. HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 CinemaIQ</h1>
    <p>AI-Powered Box Office Intelligence & Production Strategy Platform</p>
    <div class="badges">
        <span class="badge">Machine Learning</span>
        <span class="badge">South Asian Cinema</span>
        <span class="badge">Real-Time Inference</span>
        <span class="badge">ROI Optimization</span>
        <span class="badge">Scenario Analysis</span>
    </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 11. TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Prediction Engine", "Strategy Battle", "Market Insights"])

# ════════════════════════════════════════════
# TAB 1 — PREDICTION ENGINE
# ════════════════════════════════════════════
with tab1:

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns([1.2, 1, 1], gap="large")

    with r1c1:
        st.markdown("<div class='sec-title'>Production Budget</div>", unsafe_allow_html=True)
        budget = st.number_input("Budget", min_value=1_000_000, max_value=500_000_000,
                                 value=50_000_000, step=1_000_000, label_visibility="collapsed")
        st.markdown(f"<div style='color:rgba(255,255,255,.35);font-size:.8rem;margin-top:-6px;'>"
                    f"Selected: <b style='color:#FFD700;'>${budget/1e6:.1f}M</b></div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        genres = st.multiselect("Genre(s)", artifacts['mlb'].classes_, default=["Action"])

    with r1c2:
        st.markdown("<div class='sec-title'>Film Parameters</div>", unsafe_allow_html=True)
        runtime  = st.number_input("Runtime (mins)", 60, 300, 150, 1)
        vote_avg = st.number_input("Target IMDb Rating", 1.0, 10.0, 7.5, 0.1)

    with r1c3:
        st.markdown("<div class='sec-title'>Release Window</div>", unsafe_allow_html=True)
        month = st.selectbox("Release Month", range(1, 13), index=5,
                             format_func=lambda x: MONTHS[x])
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:rgba(255,215,0,.06);border:1px solid rgba(255,215,0,.15);
                    border-radius:12px;padding:14px;text-align:center;'>
            <div style='font-size:.7rem;color:rgba(255,255,255,.4);
                        text-transform:uppercase;letter-spacing:1.5px;'>Selected Window</div>
            <div style='font-size:1.4rem;font-weight:700;color:#FFD700;margin-top:6px;'>
                {MONTHS[month]}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    r2c1, r2c2, r2c3 = st.columns([1.5, 1, 1], gap="large")

    with r2c1:
        st.markdown("<div class='sec-title'>Star Cast</div>", unsafe_allow_html=True)
        actors = st.multiselect("Actors", actor_list, default=["Shah Rukh Khan"])
        if actors:
            img_cols = st.columns(min(len(actors), 5))
            for i, act in enumerate(actors[:5]):
                with img_cols[i]:
                    img = fetch_image(act)
                    pw  = artifacts['actor_power'].get(act, 0)
                    if img: st.image(img, width=60)
                    st.markdown(
                        f"<div style='font-size:.65rem;color:rgba(255,255,255,.65);text-align:center;'>"
                        f"{act.split()[0]}</div>"
                        f"<div style='text-align:center;margin-top:2px;'>"
                        f"<span class='power-tag'>⚡ {pw:.1f}</span></div>",
                        unsafe_allow_html=True)

    with r2c2:
        st.markdown("<div class='sec-title'>Director</div>", unsafe_allow_html=True)
        director = st.selectbox("Director", dir_list,
                                index=dir_list.index("S.S. Rajamouli") if "S.S. Rajamouli" in dir_list else 0)
        di = fetch_image(director)
        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            if di: st.image(di, width=60)
        with d_col2:
            st.markdown(f"""
            <div style='padding-top:6px;'>
                <div style='font-size:.8rem;color:rgba(255,255,255,.7);font-weight:600;'>{director}</div>
                <div style='margin-top:5px;'>
                    <span class='power-tag'>⚡ {artifacts['director_power'].get(director,0):.1f} · Director</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with r2c3:
        st.markdown("<div class='sec-title'>Composer</div>", unsafe_allow_html=True)
        composer = st.selectbox("Composer", comp_list,
                                index=comp_list.index("Anirudh Ravichander") if "Anirudh Ravichander" in comp_list else 0)
        ci = fetch_image(composer)
        c_col1, c_col2 = st.columns([1, 2])
        with c_col1:
            if ci: st.image(ci, width=60)
        with c_col2:
            st.markdown(f"""
            <div style='padding-top:6px;'>
                <div style='font-size:.8rem;color:rgba(255,255,255,.7);font-weight:600;'>{composer}</div>
                <div style='margin-top:5px;'>
                    <span class='power-tag'>⚡ {artifacts['composer_power'].get(composer,0):.1f} · Composer</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col2:
        run_pred = st.button("Generate Box Office Prediction", type="primary")

    # ── OUTPUT ────────────────────────────────
    if run_pred:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.spinner("Running prediction model…"):
            pess, base, opti = predict_scenarios(
                budget, runtime, vote_avg, 5000, month,
                actors, director, composer, genres)

        profit     = base - budget
        roi        = (profit / budget) * 100
        multiplier = base / budget

        profit_color = "#00CC96" if profit >= 0 else "#EF553B"
        profit_label = "✅ Projected Profit" if profit >= 0 else "❌ Projected Loss"

        st.markdown(f"""
        <div class='result-card'>
            <div class='result-label'>Predicted Box Office Revenue</div>
            <div class='result-revenue'>${base/1e6:.1f}M</div>
            <div style='color:{profit_color};font-size:1.1rem;font-weight:600;margin-top:14px;'>
                {profit_label}: ${abs(profit)/1e6:.1f}M
            </div>
            <div style='color:rgba(255,255,255,.3);font-size:.8rem;margin-top:8px;'>
                {MONTHS[month]} Release &nbsp;·&nbsp; {", ".join(genres) if genres else "Unspecified"} Genre
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # KPI Row
        k1, k2, k3 = st.columns(3)
        verdict = ("Blockbuster" if multiplier >= 3 else
                   "Hit"        if multiplier >= 2 else
                   "Borderline" if multiplier >= 1 else "Flop")
        verdict_color = ("#00CC96" if multiplier >= 2 else
                         "#FFD700" if multiplier >= 1 else "#EF553B")
        star_s = power_score(actors, artifacts['actor_power'])

        with k1:
            roi_cls = "kpi-pos" if roi >= 0 else "kpi-neg"
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-value' style='color:{"#00CC96" if roi>=0 else "#EF553B"};'>{roi:.0f}%</div>
                <div class='kpi-label'>Return on Investment</div>
                <div class='{roi_cls}'>{"Profitable" if roi>=0 else "Loss-Making"}</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-value'>{multiplier:.2f}×</div>
                <div class='kpi-label'>Revenue Multiplier</div>
                <div style='color:{verdict_color};font-size:.82rem;margin-top:5px;'>{verdict}</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-value'>{star_s:.1f}</div>
                <div class='kpi-label'>Star Power Index</div>
                <div class='kpi-gold'>⚡ Talent Score</div>
            </div>""", unsafe_allow_html=True)

        # ── Insight #1: Prediction Analysis ──
        with st.spinner("Generating AI insight…"):
            insight_1 = get_groq_insight(
                f"A South Asian film with a ${budget/1e6:.1f}M budget releasing in {MONTHS[month]}, "
                f"genres: {', '.join(genres) if genres else 'Action'}, "
                f"cast: {', '.join(actors[:3]) if actors else 'not specified'}, "
                f"director: {director}, composer: {composer}, "
                f"target IMDb rating {vote_avg}, runtime {runtime} mins. "
                f"The ML model predicts ${base/1e6:.1f}M revenue, {roi:.0f}% ROI, "
                f"verdict: {verdict}. Star power index: {star_s:.1f}. "
                f"Pessimistic: ${pess/1e6:.1f}M, Optimistic: ${opti/1e6:.1f}M. "
                f"In 2-3 sentences, explain what is driving this prediction and "
                f"the single most important thing the producer should do to improve the outcome."
            )
        show_insight(insight_1)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        ch1, ch2 = st.columns(2, gap="large")

        with ch1:
            st.markdown("<div class='sec-title'>Revenue Scenario Analysis</div>", unsafe_allow_html=True)
            fig_range = go.Figure()
            for lbl, val, clr in [("Pessimistic", pess, "#EF553B"),
                                   ("Base Case",   base, "#FFD700"),
                                   ("Optimistic",  opti, "#00CC96")]:
                fig_range.add_trace(go.Bar(
                    x=[lbl], y=[val/1e6], name=lbl,
                    marker_color=clr, marker_opacity=.85, showlegend=False,
                    text=[f"${val/1e6:.1f}M"], textposition='outside',
                    textfont=dict(color='white', size=13)))
            fig_range.add_hline(y=budget/1e6, line_dash="dash",
                                line_color="rgba(255,255,255,.35)",
                                annotation_text=f"Budget ${budget/1e6:.0f}M",
                                annotation_font_color="rgba(255,255,255,.55)")
            fig_range.update_layout(**make_layout(height=320, bargap=.35,
                                                  yaxis_title="Revenue ($M)",
                                                  yaxis_ticksuffix='M'))
            st.plotly_chart(fig_range, use_container_width=True)

        with ch2:
            st.markdown("<div class='sec-title'>Budget Sensitivity</div>", unsafe_allow_html=True)
            test_b = np.linspace(budget * .25, budget * 4, 35)
            test_r = [predict(b, runtime, vote_avg, 5000, month,
                              actors, director, composer, genres) for b in test_b]
            test_p = [r - b for r, b in zip(test_r, test_b)]

            fig_s = make_subplots(specs=[[{"secondary_y": True}]])
            fig_s.add_trace(go.Scatter(
                x=test_b/1e6, y=[r/1e6 for r in test_r], name="Revenue",
                line=dict(color='#FFD700', width=2.5),
                fill='tozeroy', fillcolor='rgba(255,215,0,.04)'), secondary_y=False)
            fig_s.add_trace(go.Scatter(
                x=test_b/1e6, y=[p/1e6 for p in test_p], name="Profit / Loss",
                line=dict(color='#00CC96', width=2, dash='dot')), secondary_y=True)
            fig_s.add_vline(x=budget/1e6, line_dash="dash",
                            line_color="rgba(255,80,80,.6)",
                            annotation_text="Current Budget",
                            annotation_font_color="rgba(255,130,130,.8)")
            fig_s.add_hline(y=0, line_color="rgba(255,255,255,.08)", secondary_y=True)
            fig_s.update_layout(**make_layout(height=320, hovermode='x unified'))
            fig_s.update_layout(legend=dict(orientation='h', y=1.15, x=.5, xanchor='center'))
            fig_s.update_xaxes(title_text="Budget ($M)", ticksuffix='M')
            fig_s.update_yaxes(title_text="Revenue ($M)", ticksuffix='M',
                               gridcolor='rgba(255,255,255,.05)', secondary_y=False)
            fig_s.update_yaxes(title_text="Profit ($M)", ticksuffix='M',
                               showgrid=False, secondary_y=True)
            st.plotly_chart(fig_s, use_container_width=True)

        # ── Insight #2: Budget Strategy ──
        breakeven_b = next((test_b[i]/1e6 for i in range(len(test_p)) if test_p[i] >= 0), None)
        with st.spinner("Analysing budget curve…"):
            insight_2 = get_groq_insight(
                f"Budget sensitivity analysis for a {', '.join(genres) if genres else 'Action'} film "
                f"with {director} directing and {', '.join(actors[:2]) if actors else 'cast'}. "
                f"Current budget ${budget/1e6:.1f}M yields ${base/1e6:.1f}M revenue. "
                f"{'Breakeven at $' + f'{breakeven_b:.1f}M budget.' if breakeven_b else 'Film does not break even in the tested range.'} "
                f"At 4x budget (${budget*4/1e6:.1f}M), revenue is ${test_r[-1]/1e6:.1f}M. "
                f"In 2-3 sentences, what does the budget curve tell us about the "
                f"optimal investment level for this particular production configuration?"
            )
        show_insight(insight_2)

    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='placeholder'>
            <div style='color:rgba(255,255,255,.25);font-size:.95rem;'>Fill in the details above</div>
            <div style='color:rgba(255,215,0,.4);font-size:.85rem;font-weight:600;'>
                then click Generate Prediction
            </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — STRATEGY BATTLE
# ════════════════════════════════════════════
with tab2:
    st.markdown("<div style='color:rgba(255,255,255,.45);margin-bottom:22px;'>"
                "Head-to-head comparison of two production strategies to find the highest ROI path.</div>",
                unsafe_allow_html=True)

    ca, cb = st.columns(2, gap="large")

    with ca:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='scenario-label-a'>SCENARIO A</div>", unsafe_allow_html=True)
        a_budget = st.number_input("Budget A", value=40_000_000, step=1_000_000, key="a_bud",
                                   label_visibility="collapsed")
        st.markdown(f"<div style='color:rgba(255,255,255,.35);font-size:.8rem;margin-bottom:10px;'>"
                    f"${a_budget/1e6:.1f}M</div>", unsafe_allow_html=True)
        a_actors = st.multiselect("Cast A", actor_list, default=["Allu Arjun"], key="a_act")
        a_dir    = st.selectbox("Director A", dir_list,
                                index=dir_list.index("Sukumar") if "Sukumar" in dir_list else 0, key="a_dir")
        a_genres = st.multiselect("Genre A", artifacts['mlb'].classes_, default=["Action"], key="a_gen")
        a_month  = st.selectbox("Release Month A", range(1,13), index=3,
                                format_func=lambda x: MONTHS[x], key="a_month")
        st.markdown("</div>", unsafe_allow_html=True)

    with cb:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='scenario-label-b'>SCENARIO B</div>", unsafe_allow_html=True)
        b_budget = st.number_input("Budget B", value=30_000_000, step=1_000_000, key="b_bud",
                                   label_visibility="collapsed")
        b_actors = st.multiselect("Cast B", actor_list, default=["Fahadh Faasil"], key="b_act")
        b_dir    = st.selectbox("Director B", dir_list,
                                index=dir_list.index("Lokesh Kanagaraj") if "Lokesh Kanagaraj" in dir_list else 0, key="b_dir")
        b_genres = st.multiselect("Genre B", artifacts['mlb'].classes_, default=["Thriller"], key="b_gen")
        b_month  = st.selectbox("Release Month B", range(1,13), index=11,
                                format_func=lambda x: MONTHS[x], key="b_month")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    btn_b1, btn_b2, btn_b3 = st.columns([1, 1, 1])
    with btn_b2:
        battle_btn = st.button("Run Strategy Battle", type="primary")

    if battle_btn:
        with st.spinner("Simulating both strategies…"):
            rev_a = predict(a_budget,150,7.5,5000,a_month,a_actors,a_dir,"Anirudh Ravichander",a_genres)
            rev_b = predict(b_budget,150,7.5,5000,b_month,b_actors,b_dir,"Anirudh Ravichander",b_genres)

        profit_a, profit_b = rev_a - a_budget, rev_b - b_budget
        roi_a,    roi_b    = (profit_a/a_budget)*100, (profit_b/b_budget)*100

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        for col, label, val, delta, is_pos in [
            (k1, "Revenue A", f"${rev_a/1e6:.1f}M",  f"${profit_a/1e6:.1f}M profit", profit_a>0),
            (k2, "Revenue B", f"${rev_b/1e6:.1f}M",  f"${profit_b/1e6:.1f}M profit", profit_b>0),
            (k3, "ROI — A",   f"{roi_a:.0f}%",        "Return on investment",          roi_a>0),
            (k4, "ROI — B",   f"{roi_b:.0f}%",        "Return on investment",          roi_b>0),
        ]:
            with col:
                dc = "#00CC96" if is_pos else "#EF553B"
                st.markdown(f"""<div class='kpi-card'>
                    <div class='kpi-value'>{val}</div>
                    <div class='kpi-label'>{label}</div>
                    <div style='color:{dc};font-size:.8rem;margin-top:5px;'>{delta}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        winner  = "A" if profit_a > profit_b else "B"
        gap     = abs(profit_a - profit_b)
        roi_gap = abs(roi_a - roi_b)
        st.markdown(f"""
        <div class='verdict verdict-win'>
            Scenario <strong>{winner}</strong> wins by
            <strong>${gap/1e6:.1f}M</strong> in profit &nbsp;|&nbsp;
            ROI gap: <strong>{roi_gap:.0f}pp</strong>
            &nbsp;({MONTHS[a_month]} vs {MONTHS[b_month]} release)
        </div>""", unsafe_allow_html=True)

        # ── Insight #3: Strategy Comparison ──
        with st.spinner("Generating strategic insight…"):
            insight_3 = get_groq_insight(
                f"Comparing two South Asian film strategies: "
                f"Scenario A — ${a_budget/1e6:.1f}M budget, cast: {', '.join(a_actors[:2]) if a_actors else 'TBD'}, "
                f"director: {a_dir}, genre: {', '.join(a_genres)}, releasing {MONTHS[a_month]}. "
                f"Predicted revenue ${rev_a/1e6:.1f}M, profit ${profit_a/1e6:.1f}M, ROI {roi_a:.0f}%. "
                f"Scenario B — ${b_budget/1e6:.1f}M budget, cast: {', '.join(b_actors[:2]) if b_actors else 'TBD'}, "
                f"director: {b_dir}, genre: {', '.join(b_genres)}, releasing {MONTHS[b_month]}. "
                f"Predicted revenue ${rev_b/1e6:.1f}M, profit ${profit_b/1e6:.1f}M, ROI {roi_b:.0f}%. "
                f"Scenario {winner} wins by ${gap/1e6:.1f}M. "
                f"In 2-3 sentences, explain why Scenario {winner} outperforms and "
                f"identify the biggest risk factor in the losing scenario."
            )
        show_insight(insight_3)

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("<div class='sec-title'>Revenue vs Budget Breakdown</div>", unsafe_allow_html=True)
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                name='Revenue', x=['Scenario A','Scenario B'],
                y=[rev_a/1e6, rev_b/1e6],
                marker_color=['rgba(0,204,150,.8)','rgba(239,85,59,.8)'],
                text=[f"${rev_a/1e6:.0f}M",f"${rev_b/1e6:.0f}M"],
                textposition='outside', textfont=dict(color='white')))
            fig_b.add_trace(go.Bar(
                name='Budget', x=['Scenario A','Scenario B'],
                y=[a_budget/1e6, b_budget/1e6],
                marker_color='rgba(255,215,0,.35)',
                marker_pattern_shape='/'))
            fig_b.update_layout(**make_layout(height=310, barmode='overlay',
                                             yaxis_title="$M", yaxis_ticksuffix='M'))
            st.plotly_chart(fig_b, use_container_width=True)

        with ch2:
            st.markdown("<div class='sec-title'>ROI Gauge Comparison</div>", unsafe_allow_html=True)
            fig_g = make_subplots(rows=1, cols=2,
                                  specs=[[{"type":"indicator"},{"type":"indicator"}]])
            for col_idx, (roi_val, lbl, clr) in enumerate([
                (roi_a, "Scenario A", "#00CC96"),
                (roi_b, "Scenario B", "#EF553B")], start=1):
                fig_g.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=roi_val,
                    number=dict(suffix="%", font=dict(color=clr, size=26, family='Inter')),
                    title=dict(text=lbl, font=dict(color='rgba(255,255,255,.5)', size=12)),
                    gauge=dict(
                        axis=dict(range=[-50,300], tickcolor='rgba(255,255,255,.15)'),
                        bar=dict(color=clr, thickness=.25),
                        bgcolor='rgba(255,255,255,.03)',
                        bordercolor='rgba(0,0,0,0)',
                        steps=[dict(range=[-50,0],   color='rgba(239,85,59,.08)'),
                               dict(range=[0,100],   color='rgba(255,215,0,.07)'),
                               dict(range=[100,300], color='rgba(0,204,150,.07)')]
                    )), row=1, col=col_idx)
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280,
                                font=dict(family='Inter'),
                                margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_g, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 — MARKET INSIGHTS
# ════════════════════════════════════════════
with tab3:
    st.markdown("<div style='color:rgba(255,255,255,.45);margin-bottom:22px;'>"
                "Model-derived market patterns to optimise release timing, genre selection, "
                "and talent decisions.</div>", unsafe_allow_html=True)

    month_revs = get_month_revenues(model, artifacts)
    genre_revs = get_genre_revenues(model, artifacts)

    r1c1, r1c2 = st.columns(2, gap="large")

    with r1c1:
        st.markdown("<div class='sec-title'>Optimal Release Month (Action · $50M)</div>",
                    unsafe_allow_html=True)
        best_m = month_revs.index(max(month_revs)) + 1
        worst_m = month_revs.index(min(month_revs)) + 1
        fig_m  = go.Figure(go.Bar(
            x=list(MONTHS.values()), y=[r/1e6 for r in month_revs],
            marker=dict(
                color=[r/1e6 for r in month_revs],
                colorscale=[[0,'rgba(239,85,59,.65)'],[.5,'rgba(255,215,0,.75)'],[1,'rgba(0,204,150,.85)']],
                showscale=False),
            text=[f"${r/1e6:.0f}M" for r in month_revs],
            textposition='outside', textfont=dict(color='white', size=10)))
        fig_m.add_annotation(
            x=MONTHS[best_m], y=max(month_revs)/1e6*1.08,
            text=f"Best: {MONTHS[best_m]}", showarrow=False,
            font=dict(color='#FFD700', size=12, family='Inter'))
        fig_m.update_layout(**make_layout(height=310, yaxis_title="Revenue ($M)",
                                         yaxis_ticksuffix='M'))
        fig_m.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_m, use_container_width=True)

    with r1c2:
        st.markdown("<div class='sec-title'>Genre Revenue Potential (Fixed Params)</div>",
                    unsafe_allow_html=True)
        sorted_g = sorted(genre_revs.items(), key=lambda x: x[1], reverse=True)
        g_names, g_vals = zip(*sorted_g)
        fig_genre = go.Figure(go.Bar(
            x=list(g_names), y=[v/1e6 for v in g_vals],
            marker=dict(
                color=[v/1e6 for v in g_vals],
                colorscale=[[0,'rgba(239,85,59,.65)'],[.5,'rgba(255,215,0,.75)'],[1,'rgba(0,204,150,.85)']],
                showscale=False),
            text=[f"${v/1e6:.0f}M" for v in g_vals],
            textposition='outside', textfont=dict(color='white', size=10)))
        fig_genre.update_layout(**make_layout(height=310, yaxis_title="Revenue ($M)",
                                             yaxis_ticksuffix='M'))
        fig_genre.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_genre, use_container_width=True)

    # ── Insight #4: Timing & Genre Strategy ──
    with st.spinner("Analysing market patterns…"):
        insight_4 = get_groq_insight(
            f"South Asian cinema market data from an ML model shows: "
            f"best release month is {MONTHS[best_m]} (${max(month_revs)/1e6:.1f}M avg revenue), "
            f"worst is {MONTHS[worst_m]} (${min(month_revs)/1e6:.1f}M). "
            f"Top genre: {g_names[0]} at ${g_vals[0]/1e6:.1f}M, "
            f"bottom genre: {g_names[-1]} at ${g_vals[-1]/1e6:.1f}M. "
            f"In 2-3 sentences, explain why this timing and genre pattern exists "
            f"in the South Asian film market and what a producer should take away from it."
        )
    show_insight(insight_4)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2, gap="large")

    with r2c1:
        st.markdown("<div class='sec-title'>Top 12 Actors by Power Index</div>",
                    unsafe_allow_html=True)
        top_a = sorted(artifacts['actor_power'].items(), key=lambda x: x[1], reverse=True)[:12]
        a_names, a_scores = zip(*top_a)
        fig_a = go.Figure(go.Bar(
            x=list(a_scores), y=list(a_names), orientation='h',
            marker=dict(color=list(a_scores),
                        colorscale=[[0,'rgba(255,107,53,.7)'],[1,'rgba(255,215,0,.9)']],
                        showscale=False),
            text=[f"{s:.1f}" for s in a_scores],
            textposition='outside', textfont=dict(color='white', size=11)))
        fig_a.update_layout(**make_layout(height=380))
        fig_a.update_yaxes(autorange='reversed', gridcolor='rgba(255,255,255,.05)')
        fig_a.update_xaxes(title_text="Power Score")
        st.plotly_chart(fig_a, use_container_width=True)

    with r2c2:
        st.markdown("<div class='sec-title'>Top 12 Directors by Power Index</div>",
                    unsafe_allow_html=True)
        top_d = sorted(artifacts['director_power'].items(), key=lambda x: x[1], reverse=True)[:12]
        d_names, d_scores = zip(*top_d)
        fig_d = go.Figure(go.Bar(
            x=list(d_scores), y=list(d_names), orientation='h',
            marker=dict(color=list(d_scores),
                        colorscale=[[0,'rgba(120,100,255,.65)'],[1,'rgba(200,155,255,.9)']],
                        showscale=False),
            text=[f"{s:.1f}" for s in d_scores],
            textposition='outside', textfont=dict(color='white', size=11)))
        fig_d.update_layout(**make_layout(height=380))
        fig_d.update_yaxes(autorange='reversed', gridcolor='rgba(255,255,255,.05)')
        fig_d.update_xaxes(title_text="Power Score")
        st.plotly_chart(fig_d, use_container_width=True)

    # ── Insight #5: Talent Pool ──
    with st.spinner("Analysing talent landscape…"):
        insight_5 = get_groq_insight(
            f"In South Asian cinema, the top actor by ML power index is {a_names[0]} ({a_scores[0]:.1f}), "
            f"followed by {a_names[1]} ({a_scores[1]:.1f}) and {a_names[2]} ({a_scores[2]:.1f}). "
            f"Top director: {d_names[0]} ({d_scores[0]:.1f}), "
            f"followed by {d_names[1]} ({d_scores[1]:.1f}). "
            f"Power index gap between rank 1 and rank 12 actor: "
            f"{a_scores[0]-a_scores[-1]:.1f} points. "
            f"In 2-3 sentences, what does this talent concentration tell us about "
            f"star dependency risk in South Asian cinema production decisions?"
        )
    show_insight(insight_5)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Model Knowledge Base</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    for col, label, val in [
        (s1, "Actors Indexed",    len(artifacts['actor_power'])),
        (s2, "Directors Indexed", len(artifacts['director_power'])),
        (s3, "Composers Indexed", len(artifacts['composer_power'])),
        (s4, "Genre Classes",     len(artifacts['mlb'].classes_)),
    ]:
        with col:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-value' style='color:#FFD700;'>{val}</div>
                <div class='kpi-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:16px;border-top:1px solid rgba(255,255,255,.05);'>
    <span style='color:rgba(255,255,255,.2);font-size:.75rem;'>
        CinemaIQ &nbsp;·&nbsp; Python · Streamlit · Plotly · Scikit-learn · Groq AI
        &nbsp;·&nbsp; Data Science Portfolio Project
    </span>
</div>""", unsafe_allow_html=True)