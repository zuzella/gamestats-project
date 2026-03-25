import streamlit as st
import pandas as pd
import base64
from datetime import datetime

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

        h1, h2, h3, p, label {{
            color: white !important;
        }}

        .stMarkdown, .stText, .stSubheader {{
            color: white !important;
        }}

        [data-baseweb="select"] > div {{
            background-color: rgba(255,255,255,0.92) !important;
            border-radius: 12px !important;
        }}

        [data-baseweb="select"] span {{
            color: black !important;
        }}

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

# ─── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/results.csv", parse_dates=["date"])
    return df

df = load_data()
teams = sorted(set(df["home_team"]).union(set(df["away_team"])))

# ─── Helper: recent form ──────────────────────────────────────────────────────
def get_recent_form(df, team, n=10):
    """
    Returns win rate of last n matches for a team.
    Considers home AND away matches.
    """
    team_matches = df[
        (df["home_team"] == team) | (df["away_team"] == team)
    ].sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        return 0.5  # neutral fallback

    wins = 0
    for _, row in team_matches.iterrows():
        if row["home_team"] == team:
            if row["home_score"] > row["away_score"]:
                wins += 1
        else:
            if row["away_score"] > row["home_score"]:
                wins += 1

    return wins / len(team_matches)


# ─── Helper: weighted H2H ────────────────────────────────────────────────────
def get_weighted_h2h(df, home_team, away_team):
    """
    Head-to-head win rates, where more recent matches have higher weight.
    Exponential decay: weight = exp(-k * years_ago), k=0.1
    Returns (home_score, away_score) as weighted fractions.
    """
    import math

    matches = df[
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    ].copy()

    if matches.empty:
        return None, None, matches

    now = pd.Timestamp(datetime.now())
    k = 0.1  # decay constant

    home_weighted = 0.0
    away_weighted = 0.0
    total_weight = 0.0

    for _, row in matches.iterrows():
        years_ago = (now - row["date"]).days / 365.25
        weight = math.exp(-k * years_ago)

        if row["home_score"] == row["away_score"]:
            home_weighted += weight * 0.5
            away_weighted += weight * 0.5
        elif row["home_team"] == home_team:
            if row["home_score"] > row["away_score"]:
                home_weighted += weight
            else:
                away_weighted += weight
        else:  # home_team played as away
            if row["away_score"] > row["home_score"]:
                home_weighted += weight
            else:
                away_weighted += weight

        total_weight += weight

    if total_weight == 0:
        return 0.5, 0.5, matches

    return home_weighted / total_weight, away_weighted / total_weight, matches


# ─── Main prediction ──────────────────────────────────────────────────────────
def predict(df, home_team, away_team):
    """
    Final score = 0.50 * forma + 0.35 * h2h_ważone + 0.15 * home_advantage
    """
    # 1. Form
    home_form = get_recent_form(df, home_team)
    away_form = get_recent_form(df, away_team)

    # normalize so they sum to 1
    form_total = home_form + away_form
    if form_total == 0:
        home_form_norm = away_form_norm = 0.5
    else:
        home_form_norm = home_form / form_total
        away_form_norm = away_form / form_total

    # 2. Weighted H2H
    h2h_home, h2h_away, matches = get_weighted_h2h(df, home_team, away_team)

    if h2h_home is None:
        # No H2H data → rely only on form + home advantage
        h2h_home = h2h_away = 0.5

    # 3. Home advantage
    home_advantage = 0.55  # statistically ~55% for home team
    away_disadvantage = 0.45

    # 4. Weighted final score
    home_score = (0.50 * home_form_norm) + (0.35 * h2h_home) + (0.15 * home_advantage)
    away_score = (0.50 * away_form_norm) + (0.35 * h2h_away) + (0.15 * away_disadvantage)

    # normalize to %
    total = home_score + away_score
    home_pct = round((home_score / total) * 100, 1)
    away_pct = round((away_score / total) * 100, 1)

    return home_pct, away_pct, home_form, away_form, matches


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("⚽ GameStats")
st.subheader("Match Winner Predictor")
st.write("Select two national teams and get a prediction based on form, head-to-head history and home advantage.")

home_team = st.selectbox("Home Team", teams)
away_team = st.selectbox("Away Team", teams)

if st.button("Predict Winner"):
    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        home_pct, away_pct, home_form, away_form, matches = predict(df, home_team, away_team)

        st.write("### Prediction")

        col1, col2 = st.columns(2)
        col1.metric(f"🏠 {home_team}", f"{home_pct}%")
        col2.metric(f"✈️ {away_team}", f"{away_pct}%")

        if home_pct > away_pct:
            st.success(f"Prediction: **{home_team}** is more likely to win.")
        elif away_pct > home_pct:
            st.success(f"Prediction: **{away_team}** is more likely to win.")
        else:
            st.warning("Prediction: this matchup looks perfectly balanced.")

        # Confidence bar
        st.progress(int(home_pct))

        # Form breakdown
        st.write("### Breakdown")
        col1, col2, col3 = st.columns(3)
        col1.metric("🏠 Recent form", f"{round(home_form*100, 1)}%", help="Win rate last 10 matches")
        col2.metric("✈️ Recent form", f"{round(away_form*100, 1)}%", help="Win rate last 10 matches")
        col3.metric("H2H matches", len(matches))

        st.caption("📊 Formula: 50% recent form + 35% weighted H2H + 15% home advantage")

        # Last 5 H2H
        if not matches.empty:
            st.write("### Last 5 Head-to-Head Matches")
            st.dataframe(
                matches[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]]
                .sort_values("date", ascending=False)
                .head(5)
                .reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.info("No previous matches found between these teams. Prediction is based on recent form only.")
