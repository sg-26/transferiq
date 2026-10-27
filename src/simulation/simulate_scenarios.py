"""
simulate_scenarios.py
=====================
Monte Carlo simulation of player value/performance trajectories.

For each player, runs N simulations of next-season outcomes:
  - Performance trajectory (goal contributions, minutes)
  - Market value evolution (growth/decline based on age, performance)
  - Risk-adjusted returns (Sharpe-like ratio)

Outputs:
  - data/processed/simulation_results.csv
  - data/processed/player_risk_profiles.csv
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

N_SIMULATIONS = 1000
RANDOM_SEED = 42


def load_data():
    features = pd.read_csv(PROCESSED_DIR / "features.csv")
    decline_pred = pd.read_csv(PROCESSED_DIR / "decline_predictions.csv")
    with open(MODEL_DIR / "sporting_risk_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return features, decline_pred, model_data


def simulate_market_value_change(
    current_value: float,
    age: float,
    decline_prob: float,
    performance_delta: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate market value changes for next season.

    Model:
      - Base growth rate depends on age (young = positive, old = negative)
      - Decline probability shifts the distribution downward
      - Performance delta (goals/assists change) impacts value
      - Add noise for market uncertainty
    """
    # Age-based base growth rate
    if age < 23:
        base_growth = 0.15  # young players appreciate
    elif age < 27:
        base_growth = 0.05  # peak years, moderate growth
    elif age < 30:
        base_growth = -0.02  # slight decline
    elif age < 33:
        base_growth = -0.10  # accelerating decline
    else:
        base_growth = -0.20  # steep decline

    # Decline probability impact
    decline_impact = -decline_prob * 0.30  # higher decline prob = lower value

    # Performance impact (normalized)
    perf_impact = np.clip(performance_delta * 0.05, -0.20, 0.30)

    # Total expected growth
    expected_growth = base_growth + decline_impact + perf_impact

    # Simulate: normal distribution around expected growth
    growth_std = 0.15 + age * 0.005  # older = more uncertain
    simulated_growth = rng.normal(expected_growth, growth_std, n_sims)

    # Apply to current value
    simulated_values = current_value * (1 + simulated_growth)

    # Floor at 0.1M (players don't go to 0, they get released)
    simulated_values = np.maximum(simulated_values, 100_000)

    return simulated_values


def simulate_performance_trajectory(
    current_gc: float,
    decline_prob: float,
    age: float,
    minutes_played: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate next-season goal contributions.

    Model:
      - Expected GC = current GC * (1 - decline_adjustment)
      - Decline probability increases the chance of significant drop
      - Age factor: older players decline faster
      - Minutes availability adds uncertainty
    """
    # Expected performance change
    decline_shift = decline_prob * 0.25
    age_shift = max(0, (age - 28) * 0.02)

    expected_gc = current_gc * (1 - decline_shift - age_shift)

    # Minutes utilization uncertainty
    minutes_factor = min(1.0, minutes_played / (90 * 30))
    expected_gc *= minutes_factor

    # Volatility increases with decline probability
    gc_std = max(1.0, current_gc * 0.30 + decline_prob * 2.0)

    simulated_gc = rng.normal(expected_gc, gc_std, n_sims)
    simulated_gc = np.maximum(simulated_gc, 0)

    return simulated_gc


def run_simulations(features: pd.DataFrame, decline_pred: pd.DataFrame) -> pd.DataFrame:
    """Run Monte Carlo simulations for all players."""
    log.info("Running %d Monte Carlo simulations per player...", N_SIMULATIONS)

    rng = np.random.default_rng(RANDOM_SEED)

    # Merge decline predictions into features
    merge_cols = ["player_id", "season_id"]
    df = features.merge(
        decline_pred[merge_cols + ["decline_probability"]],
        on=merge_cols,
        how="left",
    )
    df["decline_probability"] = df["decline_probability"].fillna(0.5)

    results = []
    for _, row in df.iterrows():
        player_id = row["player_id"]
        short_name = row.get("short_name", "")
        team_name = row.get("team_name", "")
        season_name = row.get("season_name", "")
        age = row.get("age_at_season_end", 25)
        if pd.isna(age):
            age = 25

        current_value = row.get("market_value_eur", 0)
        if pd.isna(current_value) or current_value <= 0:
            current_value = 1_000_000  # default 1M for missing

        current_gc = row.get("current_gc", 0)
        if pd.isna(current_gc):
            current_gc = 0

        decline_prob = row["decline_probability"]
        minutes_played = row.get("stat_total_minutes_played", 1000)

        # Simulate
        sim_values = simulate_market_value_change(
            current_value, age, decline_prob, 0, N_SIMULATIONS, rng
        )
        sim_gc = simulate_performance_trajectory(
            current_gc, decline_prob, age, minutes_played, N_SIMULATIONS, rng
        )

        # Compute risk metrics
        expected_value = float(np.mean(sim_values))
        median_value = float(np.median(sim_values))
        var_5 = float(np.percentile(sim_values, 5))  # 5th percentile (worst case)
        var_95 = float(np.percentile(sim_values, 95))
        value_std = float(np.std(sim_values))

        # Value at Risk (VaR): how much could we lose?
        var_loss = current_value - var_5

        # Sharpe-like ratio: (expected return - risk-free) / volatility
        expected_return = (expected_value - current_value) / current_value
        sharpe = expected_return / (value_std / current_value) if value_std > 0 else 0

        expected_gc = float(np.mean(sim_gc))
        gc_std = float(np.std(sim_gc))

        # Decline risk: probability of >20% value drop
        decline_20pct = float(np.mean(sim_values < current_value * 0.8))

        results.append({
            "player_id": player_id,
            "short_name": short_name,
            "team_name": team_name,
            "season_name": season_name,
            "age": age,
            "current_value": current_value,
            "decline_prob": decline_prob,
            "expected_value": round(expected_value),
            "median_value": round(median_value),
            "var_5pct": round(var_5),
            "var_95pct": round(var_95),
            "value_std": round(value_std),
            "var_loss": round(var_loss),
            "sharpe_ratio": round(sharpe, 3),
            "expected_gc": round(expected_gc, 1),
            "gc_std": round(gc_std, 1),
            "decline_risk_20pct": round(decline_20pct, 3),
        })

    results_df = pd.DataFrame(results)
    return results_df


def compute_risk_profiles(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player risk profiles for Pareto optimization."""
    profiles = results_df.groupby(["player_id", "short_name", "team_name"]).agg({
        "age": "first",
        "current_value": "first",
        "expected_value": "first",
        "decline_prob": "first",
        "var_5pct": "first",
        "sharpe_ratio": "first",
        "expected_gc": "first",
        "decline_risk_20pct": "first",
    }).reset_index()

    # Risk-adjusted value score
    profiles["risk_adjusted_value"] = (
        profiles["expected_value"] / (profiles["current_value"] + 1)
    )

    return profiles.sort_values("risk_adjusted_value", ascending=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features, decline_pred, model_data = load_data()
    log.info("Loaded features: %s, decline predictions: %s", features.shape, decline_pred.shape)

    results_df = run_simulations(features, decline_pred)

    results_df.to_csv(OUTPUT_DIR / "simulation_results.csv", index=False)
    log.info("Saved simulation results: %s (%d rows)", OUTPUT_DIR / "simulation_results.csv", len(results_df))

    profiles = compute_risk_profiles(results_df)
    profiles.to_csv(OUTPUT_DIR / "player_risk_profiles.csv", index=False)
    log.info("Saved risk profiles: %s (%d players)", OUTPUT_DIR / "player_risk_profiles.csv", len(profiles))

    # Summary
    log.info("\n--- Top 10 Risk-Adjusted Value ---")
    log.info(profiles.head(10)[
        ["short_name", "team_name", "age", "current_value", "expected_value",
         "sharpe_ratio", "decline_risk_20pct"]
    ].to_string(index=False))

    return results_df, profiles


if __name__ == "__main__":
    main()
