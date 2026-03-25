import streamlit as st
import pandas as pd
import pickle
import math
import base64

st.set_page_config(page_title="GameStats", page_icon="⚽", layout="centered")

def add_bg_and_style(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .main .block-container {{
            background: rgba(10, 18, 35, 0.68);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 2.2rem;
            border-radius: 22px;
            margin-top: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        }}
        h1, h2, h3, p, label {{ color: white !important; }}
        .stMarkdown, .stText, .stSubheader {{ color: white !important; }}
        [data-baseweb="select"] > div {{
            background-color: rgba(255,255,255,0.92) !important;
            border-radius: 12px !important;
        }}
        [data-baseweb="select"] span {{ color: black !important; }}
        .stButton > button {{
            background: linear-gradient(90deg, #1e90ff, #00c6ff);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }}
        .stButton > button:hover {{
            background: linear-gradient(90deg, #187bcd, #00a8dd);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_and_style("background.jpg")

# ─── Load data & model ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/results.csv", parse_dates=["date"])

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()
teams = sorted(set(df["home_team"]).union(set(df["away_team"])))

# ─── Feature helpers ──────────────────────────────────────────────────────────
def get_team_stats(df, team, n=10):
    matches = df[
        (df["home_team"] == team) | (df["away_team"] == team)
    ].sort_values("date", ascending=False).head(n)

    if matches.empty:
        return 0.5, 1.0

    wins, goals = 0, 0
    for _, row in matches.iterrows():
        if row["home_team"] == team:
            goals += row["home_score"]
            if row["home_score"] > row["away_score"]:
                wins += 1
        else:
            goals += row["away_score"]
            if row["away_score"] > row["home_score"]:
                wins += 1

    return wins / len(matches), goals / len(matches)


def get_h2h(df, home_team, away_team):
    matches = df[
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    ].sort_values("date", ascending=False)

    if matches.empty:
        return 0.5, matches

    now = pd.Timestamp("today")
    home_w, total_w = 0.0, 0.0
    for _, row in matches.iterrows():
        years_ago = (now - row["date"]).days / 365.25
        w = math.exp(-0.1 * years_ago)
        if row["home_team"] == home_team:
            if row["home_score"] > row["away_score"]:
                home_w += w
        else:
            if row["away_score"] > row["home_score"]:
                home_w += w
        total_w += w

    return (home_w / total_w if total_w > 0 else 0.5), matches


# ─── Predict ──────────────────────────────────────────────────────────────────
def predict(home_team, away_team):
    home_wr, home_gpg = get_team_stats(df, home_team)
    away_wr, away_gpg = get_team_stats(df, away_team)
    h2h_rate, matches = get_h2h(df, home_team, away_team)

    features = pd.DataFrame([{
        "home_win_rate": home_wr,
        "away_win_rate": away_wr,
        "home_goals_per_game": home_gpg,
        "away_goals_per_game": away_gpg,
        "h2h_home_rate": h2h_rate,
        "is_neutral": 0
    }])

    proba = model.predict_proba(features)[0]
    home_pct = round(proba[1] * 100, 1)
    away_pct = round(proba[0] * 100, 1)

    return home_pct, away_pct, home_wr, away_wr, home_gpg, away_gpg, matches


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("⚽ GameStats")
st.subheader("Match Winner Predictor")
st.write("Select two national teams and get an ML-powered prediction.")

home_team = st.selectbox("Home Team", teams)
away_team = st.selectbox("Away Team", teams)

if st.button("Predict Winner"):
    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        home_pct, away_pct, home_wr, away_wr, home_gpg, away_gpg, matches = predict(home_team, away_team)

        st.write("### 🤖 ML Prediction")
        col1, col2 = st.columns(2)
        col1.metric(f"🏠 {home_team}", f"{home_pct}%")
        col2.metric(f"✈️ {away_team}", f"{away_pct}%")

        if home_pct > away_pct:
            st.success(f"Prediction: **{home_team}** is more likely to win.")
        elif away_pct > home_pct:
            st.success(f"Prediction: **{away_team}** is more likely to win.")
        else:
            st.warning("Prediction: this matchup looks perfectly balanced.")

        st.progress(int(home_pct))

        st.write("### 📊 Feature Breakdown")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏠 Form", f"{round(home_wr*100,1)}%")
        col2.metric("✈️ Form", f"{round(away_wr*100,1)}%")
        col3.metric("🏠 Goals/game", round(home_gpg, 2))
        col4.metric("✈️ Goals/game", round(away_gpg, 2))

        st.caption("🤖 Powered by Random Forest — trained on form, goals/game, weighted H2H & home advantage")

        if not matches.empty:
            st.write("### Last 5 Head-to-Head Matches")
            st.dataframe(
                matches[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]]
                .head(5)
                .reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.info("No previous H2H matches found. Prediction based on recent form only.")
