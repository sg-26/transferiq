"""
sporting_risk_model.py
======================
LightGBM model predicting the probability of performance decline next season.

Target: Binary (1 = goal contributions decline next season, 0 = stable/improve)
Training: Temporal split (train on season t -> t+1 transitions)
Output: Per-player decline probabilities + model artifacts

Data leakage prevention:
  - Only use features available AT prediction time (current season stats)
  - Train on 22/23->23/24, validate on 23/24->24/25
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(r"C:\football")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_features() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "features.csv")


def build_transition_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season_id"]).copy()
    df = df.loc[:, ~df.columns.duplicated()]

    goals_col = "stat_total_goals" if "stat_total_goals" in df.columns else None
    assists_col = "stat_total_assists" if "stat_total_assists" in df.columns else None
    if goals_col and assists_col:
        df["current_gc"] = df[goals_col].fillna(0) + df[assists_col].fillna(0)
    else:
        df["current_gc"] = 0

    df["next_gc"] = df.groupby("player_id")["current_gc"].shift(-1)
    df["has_next_season"] = df["next_gc"].notna()
    df["target_decline"] = np.where(
        df["has_next_season"],
        (df["next_gc"] < df["current_gc"]).astype(int),
        np.nan,
    )
    return df


FEATURE_COLS = [
    "age_at_season_end", "height_cm", "weight_kg",
    "stat_total_goals", "stat_total_assists", "stat_total_minutes_played",
    "stat_total_shots", "stat_total_shots_on_target",
    "stat_total_passes", "stat_total_successful_passes",
    "stat_total_key_passes", "stat_total_crosses",
    "stat_total_duels", "stat_total_duels_won",
    "stat_total_defensive_actions", "stat_total_interceptions",
    "stat_total_recoveries",
    "stat_total_dribbles", "stat_total_successful_dribbles",
    "stat_total_yellow_cards", "stat_total_red_cards",
    "stat_total_fouls", "stat_total_fouls_suffered",
    "stat_percent_successful_passes_percent",
    "stat_percent_duels_won_percent",
    "stat_percent_offensive_duels_won_percent",
    "stat_percent_defensive_duels_won_percent",
    "minutes_utilization", "goal_contributions_p90",
    "shot_conversion", "shot_on_target_rate", "pass_completion",
    "duels_won_rate", "dribble_success",
    "key_passes_p90", "progressive_actions_p90",
    "age_risk", "minutes_risk", "starts_ratio",
    "card_risk", "position_risk",
    "fouls_suffered_p90",
    "win_rate_with",
    "team_league_rank_filled",
    "log_market_value",
    "matches", "matches_started",
]


def get_available_features(df: pd.DataFrame) -> list:
    return [c for c in FEATURE_COLS if c in df.columns]


def train_model(df: pd.DataFrame) -> dict:
    log.info("Building transition dataset...")
    df = df.loc[:, ~df.columns.duplicated()]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_cols = ["player_id", "season_id"]
    keep_cols = list(dict.fromkeys(id_cols + ["current_gc", "next_gc",
                      "has_next_season", "target_decline"] + numeric_cols))
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_numeric = df[keep_cols].copy()

    train_data = df_numeric[df_numeric["has_next_season"] == True].copy()
    log.info("  Transition rows: %d", len(train_data))

    decline_rate = float(train_data["target_decline"].mean()) * 100
    log.info("  Decline rate: %.1f%%", decline_rate)

    feature_cols = get_available_features(train_data)
    feature_cols = list(dict.fromkeys(feature_cols))

    usable_features = []
    for c in feature_cols:
        if c in train_data.columns and train_data[c].dtype in [np.float64, np.int64, np.float32, np.int32]:
            usable_features.append(c)

    log.info("  Usable features: %d", len(usable_features))

    train_mask = train_data["season_id"] == 188124
    test_mask = train_data["season_id"] == 189012

    X_train = train_data.loc[train_mask, usable_features].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32).values
    y_train = train_data.loc[train_mask, "target_decline"].fillna(0).astype(int).values
    X_test = train_data.loc[test_mask, usable_features].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32).values
    y_test = train_data.loc[test_mask, "target_decline"].fillna(0).astype(int).values

    log.info("  Train: %d rows, %d features, Test: %d rows",
             X_train.shape[0], X_train.shape[1], X_test.shape[0])
    log.info("  Train decline rate: %.1f%%, Test decline rate: %.1f%%",
             float(y_train.mean()) * 100, float(y_test.mean()) * 100)

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [lgb.log_evaluation(0)]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=200,
        valid_sets=[valid_set],
        callbacks=callbacks,
    )

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_pred_proba)

    log.info("  Test AUC: %.4f", auc)
    log.info("  Test Accuracy: %.4f", acc)
    log.info("  Test Log Loss: %.4f", ll)
    log.info("  Classification Report:\n%s",
             classification_report(y_test, y_pred))

    importance = pd.DataFrame({
        "feature": usable_features,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    log.info("  Top 10 features:\n%s", importance.head(10).to_string(index=False))

    return {
        "model": model,
        "feature_cols": usable_features,
        "auc": auc,
        "accuracy": acc,
        "log_loss": ll,
        "importance": importance,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
    }


def predict_decline_probabilities(df: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available_features = [c for c in feature_cols if c in numeric_cols]

    X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32).values
    df["decline_probability"] = model.predict(X)

    return df


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features()
    log.info("Loaded features: %s", df.shape)

    df = build_transition_target(df)

    result = train_model(df)

    df = predict_decline_probabilities(df, result["model"], result["feature_cols"])

    out_cols = [
        "player_id", "short_name", "team_id", "team_name",
        "season_id", "season_name", "primary_position", "age_at_season_end",
        "market_value_eur", "profitability_score", "sporting_risk_score",
        "decline_probability", "current_gc",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    predictions = df[out_cols].copy()
    predictions.to_csv(OUTPUT_DIR / "decline_predictions.csv", index=False)
    log.info("Saved decline predictions: %s", OUTPUT_DIR / "decline_predictions.csv")

    with open(MODEL_DIR / "sporting_risk_model.pkl", "wb") as f:
        pickle.dump({
            "model": result["model"],
            "feature_cols": result["feature_cols"],
            "metrics": {
                "auc": result["auc"],
                "accuracy": result["accuracy"],
                "log_loss": result["log_loss"],
            },
        }, f)
    log.info("Saved model: %s", MODEL_DIR / "sporting_risk_model.pkl")

    result["importance"].to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    return df, result


if __name__ == "__main__":
    main()
