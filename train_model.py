import pandas as pd
import numpy as np
import pickle
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/results.csv", parse_dates=["date"])

# ─── Feature helpers ──────────────────────────────────────────────────────────
def get_team_stats(df, team, before_date, n=10):
    """Win rate and avg goals for a team in last n matches before a given date."""
    matches = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    ].sort_values("date", ascending=False).head(n)

    if matches.empty:
        return 0.5, 1.0  # neutral defaults

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


def get_h2h_rate(df, home_team, away_team, before_date):
    """Weighted H2H win rate for home_team against away_team."""
    matches = df[
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    ]
    matches = matches[matches["date"] < before_date]

    if matches.empty:
        return 0.5

    home_w, total_w = 0.0, 0.0
    now = pd.Timestamp("today")
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

    return home_w / total_w if total_w > 0 else 0.5


# ─── Build feature matrix ─────────────────────────────────────────────────────
print("Building features... (this may take a few minutes)")

rows = []
# Sample every 5th row to speed up training on large dataset
sample = df.sample(frac=0.3, random_state=42)

for _, row in sample.iterrows():
    home, away, date = row["home_team"], row["away_team"], row["date"]

    home_wr, home_gpg = get_team_stats(df, home, date)
    away_wr, away_gpg = get_team_stats(df, away, date)
    h2h = get_h2h_rate(df, home, away, date)

    # target: 1 = home wins, 0 = away wins or draw
    result = 1 if row["home_score"] > row["away_score"] else 0

    rows.append({
        "home_win_rate": home_wr,
        "away_win_rate": away_wr,
        "home_goals_per_game": home_gpg,
        "away_goals_per_game": away_gpg,
        "h2h_home_rate": h2h,
        "is_neutral": int(row["neutral"]),
        "result": result
    })

features_df = pd.DataFrame(rows)

X = features_df.drop("result", axis=1)
y = features_df["result"]

# ─── Train ────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained! Accuracy: {round(acc * 100, 1)}%")

# ─── Save ─────────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl saved!")
