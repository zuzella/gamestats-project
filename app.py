import streamlit as st
import pandas as pd
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

        h1, h2, h3, p, label {{
            color: white !important;
        }}

        .stMarkdown, .stText, .stSubheader {{
            color: white !important;
        }}

        /* Dropdown background */
        [data-baseweb="select"] > div {{
            background-color: rgba(255,255,255,0.92) !important;
            border-radius: 12px !important;
        }}

        /* Dropdown text */
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

df = pd.read_csv("data/results.csv")

# Load dataset
df = pd.read_csv("data/results.csv")

# Create full team list
teams = sorted(set(df["home_team"]).union(set(df["away_team"])))

st.title("⚽ GameStats")
st.subheader("Match Winner Predictor")
st.write("Select two national teams and get a prediction based on previous match results.")

home_team = st.selectbox("Home Team", teams)
away_team = st.selectbox("Away Team", teams)

if st.button("Predict Winner"):
    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        matches = df[
            ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
            ((df["home_team"] == away_team) & (df["away_team"] == home_team))
        ]

        if matches.empty:
            st.error("No previous matches found between these teams.")
        else:
            home_team_wins = 0
            away_team_wins = 0
            draws = 0

            for _, row in matches.iterrows():
                if row["home_score"] == row["away_score"]:
                    draws += 1
                elif row["home_team"] == home_team and row["home_score"] > row["away_score"]:
                    home_team_wins += 1
                elif row["away_team"] == home_team and row["away_score"] > row["home_score"]:
                    home_team_wins += 1
                else:
                    away_team_wins += 1

            total_matches = len(matches)

            st.write("### Historical Head-to-Head Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric(f"{home_team} wins", home_team_wins)
            col2.metric(f"{away_team} wins", away_team_wins)
            col3.metric("Draws", draws)

            st.write(f"**Total matches found:** {total_matches}")

            home_percent = round((home_team_wins / total_matches) * 100, 1)
            away_percent = round((away_team_wins / total_matches) * 100, 1)
            draw_percent = round((draws / total_matches) * 100, 1)

            st.write("### Outcome Distribution")
            st.write(f"**{home_team}:** {home_percent}%")
            st.write(f"**{away_team}:** {away_percent}%")
            st.write(f"**Draw:** {draw_percent}%")

            if home_team_wins > away_team_wins:
                st.success(f"Prediction: **{home_team}** is more likely to win.")
                st.info(f"Estimated confidence: **{home_percent}%**")
            elif away_team_wins > home_team_wins:
                st.success(f"Prediction: **{away_team}** is more likely to win.")
                st.info(f"Estimated confidence: **{away_percent}%**")
            else:
                st.warning("Prediction: this matchup looks very balanced.")
                st.info("Estimated confidence: **50%**")

            st.write("### Last 5 Matches Between Teams")
            st.dataframe(
                matches[["date", "home_team", "away_team", "home_score", "away_score"]]
                .sort_values(by="date", ascending=False)
                .head(5),
                use_container_width=True
            )