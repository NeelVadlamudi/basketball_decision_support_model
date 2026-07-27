from __future__ import annotations

"""Basketball analytics work sample for lineup and role-fit decision support.

The script generates synthetic possession-level data, trains a possession-value
model, validates it on a later game window, and writes stakeholder-facing
outputs for lineup review, tactical context, role fit, and model limitations.
"""

import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
DB_PATH = OUTPUT_DIR / "basketball_decision_support.db"
RANDOM_SEED = 42


@dataclass(frozen=True)
class LineupProfile:
    lineup_id: str
    lineup_profile: str
    spacing: float
    creation: float
    rim_pressure: float
    defense: float
    rebounding: float
    ball_security: float
    pace: float


LINEUPS = [
    LineupProfile("L01", "Spacing with two-way wings", 0.87, 0.72, 0.61, 0.78, 0.58, 0.74, 0.69),
    LineupProfile("L02", "High creation guard unit", 0.73, 0.88, 0.67, 0.57, 0.47, 0.62, 0.76),
    LineupProfile("L03", "Big frontcourt rebound unit", 0.55, 0.58, 0.76, 0.74, 0.88, 0.65, 0.54),
    LineupProfile("L04", "Switch-heavy defensive group", 0.64, 0.61, 0.57, 0.91, 0.68, 0.70, 0.61),
    LineupProfile("L05", "Bench spacing development group", 0.81, 0.54, 0.49, 0.52, 0.43, 0.58, 0.71),
    LineupProfile("L06", "Balanced playoff-style unit", 0.76, 0.75, 0.66, 0.79, 0.66, 0.78, 0.63),
    LineupProfile("L07", "Rim pressure transition group", 0.60, 0.70, 0.89, 0.60, 0.61, 0.57, 0.83),
    LineupProfile("L08", "Low-turnover veteran group", 0.69, 0.68, 0.55, 0.69, 0.59, 0.91, 0.52),
    LineupProfile("L09", "Small-ball offensive group", 0.91, 0.76, 0.54, 0.48, 0.36, 0.69, 0.79),
    LineupProfile("L10", "Player-development evaluation group", 0.62, 0.49, 0.58, 0.50, 0.52, 0.54, 0.67),
    LineupProfile("L11", "Defensive glass control group", 0.58, 0.57, 0.62, 0.86, 0.92, 0.69, 0.49),
    LineupProfile("L12", "Late-clock creation group", 0.74, 0.91, 0.59, 0.63, 0.51, 0.66, 0.57),
]


PLAYER_ARCHETYPES = [
    ("P01", "primary creator", 0.72, 0.95, 0.66, 0.57, 0.45, 0.67),
    ("P02", "movement shooter", 0.94, 0.62, 0.42, 0.60, 0.38, 0.73),
    ("P03", "two-way wing", 0.81, 0.69, 0.58, 0.86, 0.63, 0.71),
    ("P04", "rim protector", 0.42, 0.45, 0.75, 0.93, 0.91, 0.65),
    ("P05", "connector guard", 0.78, 0.68, 0.50, 0.70, 0.47, 0.89),
    ("P06", "bench scorer", 0.69, 0.82, 0.61, 0.42, 0.34, 0.55),
    ("P07", "defensive specialist", 0.56, 0.42, 0.48, 0.95, 0.67, 0.70),
    ("P08", "stretch big", 0.84, 0.51, 0.58, 0.62, 0.78, 0.68),
    ("P09", "transition slasher", 0.58, 0.66, 0.91, 0.61, 0.56, 0.57),
    ("P10", "development wing", 0.63, 0.48, 0.56, 0.54, 0.49, 0.50),
    ("P11", "rebounding forward", 0.52, 0.50, 0.70, 0.77, 0.94, 0.62),
    ("P12", "late-clock shot maker", 0.76, 0.88, 0.52, 0.55, 0.39, 0.59),
]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def make_possessions(n_games: int = 82, possessions_per_game: int = 54) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, float | int | str]] = []
    tactics = ["spread_pnr", "early_drag", "horns", "empty_corner", "switch_hunt", "post_split"]
    opponent_styles = ["drop", "switch", "zone", "pressure", "size", "pace"]

    for game_id in range(1, n_games + 1):
        opponent_strength = rng.normal(0.0, 0.55)
        schedule_fatigue = min(1.0, max(0.0, rng.beta(2.0, 5.5) + (0.14 if game_id % 7 in {0, 1} else 0.0)))
        for possession_number in range(possessions_per_game):
            lineup = rng.choice(LINEUPS, p=np.array([0.10, 0.09, 0.07, 0.08, 0.08, 0.12, 0.08, 0.08, 0.09, 0.07, 0.07, 0.07]))
            tactic = rng.choice(tactics)
            opponent_style = rng.choice(opponent_styles)

            late_clock = float(rng.random() < (0.14 + 0.10 * (1 - lineup.ball_security)))
            transition = float(rng.random() < (0.17 + 0.18 * lineup.pace))
            clutch_context = float(game_id > 66 and rng.random() < 0.20)

            tactic_bonus = {
                "spread_pnr": 0.08 * lineup.spacing + 0.05 * lineup.creation,
                "early_drag": 0.09 * lineup.pace + 0.05 * lineup.rim_pressure,
                "horns": 0.04 * lineup.creation + 0.03 * lineup.rebounding,
                "empty_corner": 0.07 * lineup.rim_pressure + 0.04 * lineup.spacing,
                "switch_hunt": 0.10 * lineup.creation - 0.03 * schedule_fatigue,
                "post_split": 0.06 * lineup.rebounding + 0.03 * lineup.spacing,
            }[tactic]

            style_penalty = {
                "drop": -0.02 + 0.05 * lineup.spacing,
                "switch": 0.03 * lineup.creation - 0.02 * lineup.rim_pressure,
                "zone": 0.05 * lineup.spacing - 0.03 * lineup.ball_security,
                "pressure": 0.06 * lineup.ball_security - 0.04 * schedule_fatigue,
                "size": 0.05 * lineup.rebounding - 0.03 * lineup.rim_pressure,
                "pace": 0.04 * lineup.pace - 0.03 * lineup.defense,
            }[opponent_style]

            shot_quality = np.clip(
                0.30
                + 0.28 * lineup.spacing
                + 0.20 * lineup.creation
                + 0.16 * lineup.rim_pressure
                + 0.08 * transition
                - 0.15 * late_clock
                + rng.normal(0, 0.06),
                0.05,
                0.95,
            )
            turnover_pressure = np.clip(
                0.54
                - 0.34 * lineup.ball_security
                + 0.12 * schedule_fatigue
                + 0.10 * late_clock
                + 0.08 * max(0, opponent_strength)
                + rng.normal(0, 0.05),
                0.03,
                0.90,
            )
            defensive_risk = np.clip(
                0.62
                - 0.36 * lineup.defense
                - 0.14 * lineup.rebounding
                + 0.12 * lineup.pace
                + 0.08 * schedule_fatigue
                + rng.normal(0, 0.05),
                0.03,
                0.92,
            )
            rebound_matchup = np.clip(0.18 + 0.70 * lineup.rebounding - 0.12 * opponent_strength + rng.normal(0, 0.06), 0.02, 0.98)

            true_ppp = (
                0.72
                + 0.56 * shot_quality
                - 0.42 * turnover_pressure
                - 0.21 * defensive_risk
                + 0.15 * rebound_matchup
                + tactic_bonus
                + style_penalty
                - 0.11 * schedule_fatigue
                + 0.04 * clutch_context * lineup.creation
            )
            points = np.clip(rng.normal(true_ppp, 0.34), 0, 3.2)

            rows.append(
                {
                    "game_id": game_id,
                    "possession_number": possession_number,
                    "lineup_id": lineup.lineup_id,
                    "lineup_profile": lineup.lineup_profile,
                    "tactic": tactic,
                    "opponent_style": opponent_style,
                    "opponent_strength": opponent_strength,
                    "schedule_fatigue": schedule_fatigue,
                    "late_clock": late_clock,
                    "transition": transition,
                    "clutch_context": clutch_context,
                    "spacing": lineup.spacing,
                    "creation": lineup.creation,
                    "rim_pressure": lineup.rim_pressure,
                    "defense": lineup.defense,
                    "rebounding": lineup.rebounding,
                    "ball_security": lineup.ball_security,
                    "pace": lineup.pace,
                    "shot_quality": shot_quality,
                    "turnover_pressure": turnover_pressure,
                    "defensive_risk": defensive_risk,
                    "rebound_matchup": rebound_matchup,
                    "actual_points": points,
                }
            )
    return pd.DataFrame(rows)


def make_player_roles() -> pd.DataFrame:
    rows = []
    for player_id, archetype, spacing, creation, rim_pressure, defense, rebounding, ball_security in PLAYER_ARCHETYPES:
        role_fit = (
            0.22 * spacing
            + 0.20 * creation
            + 0.16 * rim_pressure
            + 0.22 * defense
            + 0.10 * rebounding
            + 0.10 * ball_security
        )
        usage_note = (
            "high-leverage core profile"
            if role_fit >= 0.74
            else "context-dependent rotation profile"
            if role_fit >= 0.62
            else "development or matchup-specific profile"
        )
        rows.append(
            {
                "player_id": player_id,
                "archetype": archetype,
                "spacing": spacing,
                "creation": creation,
                "rim_pressure": rim_pressure,
                "defense": defense,
                "rebounding": rebounding,
                "ball_security": ball_security,
                "role_fit_score": role_fit,
                "usage_note": usage_note,
            }
        )
    return pd.DataFrame(rows).sort_values("role_fit_score", ascending=False)


def encode_features(train: pd.DataFrame, test: pd.DataFrame, features: list[str], categorical: list[str]):
    numeric = [feature for feature in features if feature not in categorical]
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    train_num = scaler.fit_transform(train[numeric])
    test_num = scaler.transform(test[numeric])
    train_cat = encoder.fit_transform(train[categorical])
    test_cat = encoder.transform(test[categorical])

    feature_names = numeric + list(encoder.get_feature_names_out(categorical))
    return np.hstack([train_num, train_cat]), np.hstack([test_num, test_cat]), feature_names


def bootstrap_lineup_intervals(df: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 7)
    rows = []
    for lineup_id, group in df.groupby("lineup_id"):
        values = group["predicted_points"].to_numpy()
        boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
        rows.append(
            {
                "lineup_id": lineup_id,
                "predicted_ppp_low": float(np.percentile(boot_means, 5)),
                "predicted_ppp_high": float(np.percentile(boot_means, 95)),
            }
        )
    return pd.DataFrame(rows)


def lineup_decision_board(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        predictions.groupby(["lineup_id", "lineup_profile"], as_index=False)
        .agg(
            possessions=("actual_points", "size"),
            actual_ppp=("actual_points", "mean"),
            predicted_ppp=("predicted_points", "mean"),
            avg_defensive_risk=("defensive_risk", "mean"),
            avg_shot_quality=("shot_quality", "mean"),
            avg_turnover_pressure=("turnover_pressure", "mean"),
            avg_rebound_matchup=("rebound_matchup", "mean"),
        )
        .merge(bootstrap_lineup_intervals(predictions), on="lineup_id", how="left")
    )

    league_avg = grouped["predicted_ppp"].mean()
    sample_factor = np.minimum(1.0, np.sqrt(grouped["possessions"] / 360.0))
    grouped["net_rating_per_100"] = (grouped["predicted_ppp"] - league_avg) * 100 * sample_factor
    grouped["net_rating_low_per_100"] = (grouped["predicted_ppp_low"] - league_avg) * 100 * sample_factor
    grouped["net_rating_high_per_100"] = (grouped["predicted_ppp_high"] - league_avg) * 100 * sample_factor
    grouped["uncertainty_width"] = (grouped["predicted_ppp_high"] - grouped["predicted_ppp_low"]) * 100
    grouped["decision_tier"] = np.select(
        [
            (grouped["net_rating_per_100"] >= 4.5) & (grouped["uncertainty_width"] <= 5.5),
            grouped["net_rating_per_100"] >= 2.0,
            grouped["net_rating_per_100"] <= -2.0,
        ],
        ["lean into", "test more", "avoid unless matchup-specific"],
        default="situational",
    )
    grouped["basketball_note"] = np.select(
        [
            grouped["avg_shot_quality"] >= 0.72,
            grouped["avg_turnover_pressure"] >= 0.36,
            grouped["avg_defensive_risk"] >= 0.38,
            grouped["avg_rebound_matchup"] >= 0.72,
        ],
        [
            "strong shot-quality profile",
            "protect with simpler actions",
            "watch transition and matchup exposure",
            "glass-control value",
        ],
        default="balanced profile",
    )

    numeric_cols = [
        "actual_ppp",
        "predicted_ppp",
        "predicted_ppp_low",
        "predicted_ppp_high",
        "avg_defensive_risk",
        "avg_shot_quality",
        "avg_turnover_pressure",
        "avg_rebound_matchup",
        "net_rating_per_100",
        "net_rating_low_per_100",
        "net_rating_high_per_100",
        "uncertainty_width",
    ]
    grouped[numeric_cols] = grouped[numeric_cols].round(3)
    return grouped.sort_values(["decision_tier", "net_rating_per_100"], ascending=[True, False])


def save_database(possessions: pd.DataFrame, predictions: pd.DataFrame, players: pd.DataFrame) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        possessions.to_sql("possessions", conn, index=False)
        predictions.to_sql("possession_predictions", conn, index=False)
        players.to_sql("player_roles", conn, index=False)


def plot_lineups(board: pd.DataFrame) -> None:
    chart = board.sort_values("net_rating_per_100", ascending=True).tail(8)
    y_pos = np.arange(len(chart))
    lower_err = chart["net_rating_per_100"] - chart["net_rating_low_per_100"]
    upper_err = chart["net_rating_high_per_100"] - chart["net_rating_per_100"]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    colors = ["#1f77b4" if tier == "lean into" else "#ff7f0e" if tier == "test more" else "#777777" for tier in chart["decision_tier"]]
    ax.barh(y_pos, chart["net_rating_per_100"], color=colors, alpha=0.88)
    ax.errorbar(
        chart["net_rating_per_100"],
        y_pos,
        xerr=np.vstack([lower_err, upper_err]),
        fmt="none",
        ecolor="#222222",
        capsize=3,
        linewidth=1.1,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(chart["lineup_profile"])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Predicted net rating vs model average per 100 possessions")
    ax.set_title("Lineup Decision Board With Uncertainty")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lineup_net_rating_uncertainty.png", dpi=180)
    plt.close(fig)


def plot_players(players: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6))
    scatter = ax.scatter(
        players["spacing"],
        players["defense"],
        s=80 + players["role_fit_score"] * 220,
        c=players["creation"],
        cmap="viridis",
        alpha=0.84,
        edgecolor="#222222",
        linewidth=0.5,
    )
    for _, row in players.iterrows():
        ax.text(row["spacing"] + 0.008, row["defense"] + 0.006, row["player_id"], fontsize=8)
    ax.set_xlabel("Spacing")
    ax.set_ylabel("Defensive versatility")
    ax.set_title("Player Role Fit Matrix")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Creation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "player_role_fit_matrix.png", dpi=180)
    plt.close(fig)


def plot_feature_importance(model: RandomForestRegressor, feature_names: list[str]) -> None:
    coefs = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    top = coefs.sort_values("importance", ascending=False).head(12).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(top["feature"], top["importance"], color="#2ca02c", alpha=0.86)
    ax.set_xlabel("Model feature importance")
    ax.set_title("Tactical and Context Signals Driving Possession Value")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tactical_feature_importance.png", dpi=180)
    plt.close(fig)


def write_markdown_outputs(metrics: dict[str, float], board: pd.DataFrame) -> None:
    top = board.sort_values("net_rating_per_100", ascending=False).head(3)
    watch = board.sort_values("avg_defensive_risk", ascending=False).head(2)

    validation = f"""# Model Validation Report

## Data Design

The dataset contains synthetic possession-level observations across game windows, lineup profiles, tactical context, opponent styles, and basketball features. It is intentionally shaped like internal basketball analytics data but does not contain any proprietary team information.

## Validation Method

- Train period: games 1 through 64
- Test period: games 65 through 82
- Target: possession points
- Model: Random forest regression with numeric scaling and one-hot encoded tactical context
- Leakage control: time-based split by game window

## Test Metrics

- Mean absolute error: {metrics['mae']:.3f} points per possession
- Root mean squared error: {metrics['rmse']:.3f} points per possession
- R-squared: {metrics['r2']:.3f}

## Interpretation

Single-possession scoring is intentionally noisy, so the model is not used as an exact possession predictor. The useful basketball output is the aggregate decision board: lineup ranking, uncertainty bands, sample-size awareness, and context notes that can guide film review, scouting discussion, or a controlled minutes test.

## Quality Checks

- All generated possessions include lineup, tactic, opponent style, and context fields
- Predicted possession values are bounded by observed basketball scoring ranges
- Lineup recommendations include sample size and bootstrap uncertainty bands
- Recommendations are not based only on raw average points

## Limitations

- Synthetic data cannot replace team tracking, scouting, medical, player-development, or coaching context
- The model is meant to demonstrate workflow design, not produce real Phoenix Suns decisions
- A production version should include possession video tags, matchup context, player availability, tracking-derived spacing, and coach-reviewed tactical labels
"""

    executive = "# Executive Brief\n\n"
    executive += "## Recommendation\n\n"
    executive += "Use the decision board as a first-pass filter, not as an automatic answer. The strongest lineups combine positive expected possession value with low uncertainty and interpretable basketball reasons.\n\n"
    executive += "## Strongest Current Tests\n\n"
    for _, row in top.iterrows():
        executive += (
            f"- {row['lineup_profile']}: {row['net_rating_per_100']:.1f} predicted net rating per 100, "
            f"{row['possessions']} possessions, tier = {row['decision_tier']}, note = {row['basketball_note']}.\n"
        )
    executive += "\n## Risk Watch\n\n"
    for _, row in watch.iterrows():
        executive += (
            f"- {row['lineup_profile']}: defensive risk {row['avg_defensive_risk']:.3f}; "
            f"use matchup review before increasing minutes.\n"
        )
    executive += "\n## Basketball Translation\n\n"
    executive += "The useful output is not just a score. It is a conversation starter for coaching, scouting, player development, and strategy: which profiles deserve more minutes, which need simpler actions, and where the model is not confident enough yet.\n"

    (OUTPUT_DIR / "model_validation.md").write_text(validation, encoding="utf-8")
    (OUTPUT_DIR / "executive_brief.md").write_text(executive, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    possessions = make_possessions()
    players = make_player_roles()

    train = possessions[possessions["game_id"] <= 64].copy()
    test = possessions[possessions["game_id"] > 64].copy()
    features = [
        "opponent_strength",
        "schedule_fatigue",
        "late_clock",
        "transition",
        "clutch_context",
        "spacing",
        "creation",
        "rim_pressure",
        "defense",
        "rebounding",
        "ball_security",
        "pace",
        "shot_quality",
        "turnover_pressure",
        "defensive_risk",
        "rebound_matchup",
        "tactic",
        "opponent_style",
    ]
    categorical = ["tactic", "opponent_style"]

    x_train, x_test, feature_names = encode_features(train, test, features, categorical)
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=12,
        max_depth=8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(x_train, train["actual_points"])
    test["predicted_points"] = model.predict(x_test)

    mse = mean_squared_error(test["actual_points"], test["predicted_points"])
    metrics = {
        "mae": mean_absolute_error(test["actual_points"], test["predicted_points"]),
        "rmse": math.sqrt(mse),
        "r2": r2_score(test["actual_points"], test["predicted_points"]),
    }

    board = lineup_decision_board(test)
    possessions.to_csv(OUTPUT_DIR / "synthetic_possessions.csv", index=False)
    test.to_csv(OUTPUT_DIR / "possession_predictions.csv", index=False)
    board.to_csv(OUTPUT_DIR / "lineup_decision_board.csv", index=False)
    players.to_csv(OUTPUT_DIR / "player_role_fit.csv", index=False)
    save_database(possessions, test, players)
    plot_lineups(board)
    plot_players(players)
    plot_feature_importance(model, feature_names)
    write_markdown_outputs(metrics, board)

    print("Basketball decision-support model complete.")
    print(f"Possessions: {len(possessions):,}")
    print(f"Validation MAE: {metrics['mae']:.3f}")
    print(f"Validation RMSE: {metrics['rmse']:.3f}")
    print(f"Validation R2: {metrics['r2']:.3f}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
