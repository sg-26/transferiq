"""
scenario_pareto.py
===================
Scenario-level Pareto frontier, now driven by the Club-Level Sporting Model.

For every valid sale scenario S_j we now APPLY the trained club sporting
predictor to the counterfactual squad:

    baseline_features (full squad)  -> xP_baseline
    post_features (post-departure)  -> xP_after
    predicted_points_delta = xP_after - xP_baseline
    predicted_relegation_risk_change = P(rel | xP_after) - P(rel | xP_baseline)

Strategic decision axes for the Pareto frontier:
    Y (Financial Return, MAX): realized sale revenue (raise_value, EUR m)
    X (Sporting Cost, MIN):    -predicted_points_delta  (points lost) or
                               predicted_relegation_risk_change

Inputs (from src/simulation/squad_counterfactual.py):
    data/processed/scenarios.csv          (has base_* / post_* capacity features)
    data/processed/scenario_players.csv
Model:
    models/club_sporting_model.pkl

Outputs:
    data/processed/scenario_pareto.csv
    data/processed/pareto_frontier_scenarios.csv
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.club_sporting_model import ClubSportingModel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(r"C:\football")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# Model-input feature names (must match squad_counterfactual base_* / post_*)
CAPACITY_FEATURES = [
    "total_goal_contributions", "total_shots", "total_assists",
    "total_market_value", "total_interceptions", "count_DF",
    "squad_minutes_utilization", "count_GK", "avg_age",
]


def load_model():
    with open(MODEL_DIR / "club_sporting_model.pkl", "rb") as f:
        d = pickle.load(f)
    return d["model"]


def apply_model_to_scenarios(df: pd.DataFrame, model: ClubSportingModel) -> pd.DataFrame:
    """Add predicted points/relegation-risk deltas to every scenario."""
    base_rows = []
    post_rows = []
    for _, r in df.iterrows():
        base = {k: (r.get(f"base_{k}", 0) or 0) for k in CAPACITY_FEATURES}
        post = {k: (r.get(f"post_{k}", 0) or 0) for k in CAPACITY_FEATURES}
        base_rows.append(base)
        post_rows.append(post)

    preds = [
        model.predict_scenario(b, p)
        for b, p in zip(base_rows, post_rows)
    ]
    pred_df = pd.DataFrame(preds)
    return pd.concat([df, pred_df], axis=1)


def pareto_optimal(df: pd.DataFrame, max_col: str, min_col: str) -> pd.Series:
    vals_max = df[max_col].values
    vals_min = df[min_col].values
    n = len(df)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isnan(vals_max[i]) or np.isnan(vals_min[i]):
            dominated[i] = True
            continue
        for j in range(n):
            if i == j or np.isnan(vals_max[j]) or np.isnan(vals_min[j]):
                continue
            dominates = (vals_max[j] > vals_max[i] and vals_min[j] <= vals_min[i]) \
                     or (vals_max[j] >= vals_max[i] and vals_min[j] < vals_min[i])
            if dominates:
                dominated[i] = True
                break
    return ~dominated


def main(team_id: int = 675, season_id: int = 189974,
         target_raise: float = 20_000_000, pareto_max_col: str = "raise_value",
         pareto_min_col: str = "predicted_points_delta"):
    model = load_model()
    log.info("Loaded ClubSportingModel (reference season %s)", model.season_id)

    df = pd.read_csv(PROCESSED_DIR / "scenarios.csv")
    log.info("Loaded %d scenarios", len(df))

    # Apply the sporting predictor to each scenario
    df = apply_model_to_scenarios(df, model)
    # Sport cost is the NEGATIVE of points delta (higher = costlier)
    df["sporting_cost"] = -df["predicted_points_delta"]
    # Revenue in EUR millions for readability
    df["revenue_eur_m"] = df["raise_value"] / 1_000_000
    df["relegation_risk_change_pct"] = df["predicted_relegation_risk_change"] * 100

    # Pareto over strategic axes: max revenue, min sporting cost
    df["pareto_optimal"] = pareto_optimal(df, pareto_max_col, pareto_min_col)
    n_pareto = int(df["pareto_optimal"].sum())
    log.info("Pareto-optimal on (max %s, min %s): %d / %d",
             pareto_max_col, pareto_min_col, n_pareto, len(df))

    df.to_csv(PROCESSED_DIR / "scenario_pareto.csv", index=False)

    pareto = df[df["pareto_optimal"]].sort_values(pareto_max_col, ascending=False)
    pareto.to_csv(PROCESSED_DIR / "pareto_frontier_scenarios.csv", index=False)

    log.info("\n--- Scenario Pareto Frontier (max revenue, min points lost) ---")
    cols = ["scenario_id", "n_players_sold", "revenue_eur_m", "raise_excess",
            "predicted_points_after", "predicted_points_delta", "sporting_cost",
            "predicted_relegation_risk_change", "relegation_risk_change_pct",
            "expected_gc_retained_pct", "depth_ok"]
    if len(pareto):
        log.info(pareto[cols].to_string(index=False))
    else:
        log.warning("(empty frontier)")

    # --- Decision-relevant view: minimum sporting cost that still clears target ---
    clears = df[df["raise_value"] >= target_raise].copy()
    if len(clears):
        clears = clears.sort_values("sporting_cost")  # min sporting cost first
        log.info("\n--- Safest way to raise >= %.0fM (min sporting cost) ---",
                 target_raise / 1e6)
        log.info(clears.head(5)[[
            "scenario_id", "n_players_sold", "revenue_eur_m", "raise_excess",
            "predicted_points_delta", "sporting_cost", "relegation_risk_change_pct",
            "expected_gc_retained_pct"]].to_string(index=False))

    return df, pareto


if __name__ == "__main__":
    main()
