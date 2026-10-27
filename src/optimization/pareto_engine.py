"""
pareto_engine.py
================
Pareto frontier optimization for Profitability vs. Sporting Risk.

Finds the set of Pareto-optimal players where no player can improve one
objective without worsening the other.

Inputs:
  - data/processed/features.csv
  - data/processed/player_risk_profiles.csv

Outputs:
  - data/processed/pareto_frontier.csv
  - data/processed/pareto_analysis.csv
  - outputs/figures/pareto_frontier.png (if matplotlib available)
"""

import logging
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_data():
    features = pd.read_csv(PROCESSED_DIR / "features.csv")
    profiles = pd.read_csv(PROCESSED_DIR / "player_risk_profiles.csv")
    decline_pred = pd.read_csv(PROCESSED_DIR / "decline_predictions.csv")
    return features, profiles, decline_pred


def find_pareto_frontier(
    costs: np.ndarray,
    values: np.ndarray,
    epsilon: float = 0.0,
) -> np.ndarray:
    """
    Find Pareto-optimal points.

    Pareto frontier: set of points where no other point has both
    lower cost AND higher value.

    Parameters:
      costs: array of cost values (lower = better)
      values: array of value values (higher = better)
      epsilon: tolerance for near-Pareto (0 = strict Pareto)

    Returns:
      Boolean array of Pareto-optimal points
    """
    n = len(costs)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has <= cost AND >= value (with epsilon tolerance)
            if costs[j] <= costs[i] + epsilon and values[j] >= values[i] - epsilon:
                if costs[j] < costs[i] - epsilon or values[j] > values[i] + epsilon:
                    is_pareto[i] = False
                    break

    return is_pareto


def run_pareto_optimization(features: pd.DataFrame, profiles: pd.DataFrame, decline_pred: pd.DataFrame) -> pd.DataFrame:
    """Find Pareto-optimal players on Profitability vs. Sporting Risk axes."""
    log.info("Running Pareto optimization...")

    # Merge decline predictions into features
    merge_cols = ["player_id", "season_id"]
    features = features.merge(
        decline_pred[merge_cols + ["decline_probability"]],
        on=merge_cols,
        how="left",
    )

    # Merge profiles with latest-season features
    latest_season = features["season_id"].max()
    latest_features = features[features["season_id"] == latest_season].copy()

    # Aggregate per player (take latest season)
    player_agg = latest_features.groupby("player_id").agg({
        "short_name": "first",
        "team_name": "first",
        "primary_position": "first",
        "age_at_season_end": "first",
        "market_value_eur": "first",
        "profitability_score": "mean",
        "sporting_risk_score": "mean",
        "decline_probability": "first",
        "goal_contributions_p90": "first",
        "stat_total_goals": "first",
        "stat_total_assists": "first",
    }).reset_index()

    # Merge with simulation profiles
    if "player_id" in profiles.columns:
        sim_cols = ["player_id", "expected_value", "sharpe_ratio",
                     "var_5pct", "decline_risk_20pct"]
        sim_cols = [c for c in sim_cols if c in profiles.columns]
        player_agg = player_agg.merge(profiles[sim_cols], on="player_id", how="left")

    # Drop players without valid scores
    valid = player_agg.dropna(subset=["profitability_score", "sporting_risk_score"]).copy()
    log.info("  Valid players for Pareto: %d", len(valid))

    # Pareto axes:
    #   X-axis (cost): sporting_risk_score (lower = better, so we MINIMIZE)
    #   Y-axis (value): profitability_score (higher = better, so we MAXIMIZE)
    costs = valid["sporting_risk_score"].values
    values = valid["profitability_score"].values

    # Use 2% epsilon for near-Pareto (makes frontier more inclusive)
    score_range = max(costs.max() - costs.min(), values.max() - values.min())
    epsilon = score_range * 0.02

    pareto_mask = find_pareto_frontier(costs, values, epsilon=epsilon)
    valid["is_pareto_optimal"] = pareto_mask

    frontier = valid[valid["is_pareto_optimal"]].copy()
    frontier = frontier.sort_values("profitability_score", ascending=False)

    log.info("  Pareto-optimal players: %d / %d", len(frontier), len(valid))

    return valid, frontier


def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Classify players into sell/hold/develop tiers based on Pareto position."""
    df = df.copy()

    # Tier logic:
    # - "SELL" if Pareto-optimal AND high profitability (top 40%) AND low risk (bottom 40%)
    # - "HOLD" if Pareto-optimal but mixed scores
    # - "DEVELOP" if not Pareto-optimal but promising (high profitability, moderate risk)
    # - "RELEASE" if not Pareto-optimal AND low profitability AND high risk

    high_profit = df["profitability_score"] >= df["profitability_score"].quantile(0.6)
    low_risk = df["sporting_risk_score"] <= df["sporting_risk_score"].quantile(0.4)

    conditions = [
        df["is_pareto_optimal"] & high_profit & low_risk,   # SELL
        df["is_pareto_optimal"] & ~low_risk,                 # HOLD
        df["is_pareto_optimal"],                              # HOLD
        ~df["is_pareto_optimal"] & high_profit,              # DEVELOP
    ]
    choices = ["SELL", "HOLD", "HOLD", "DEVELOP"]
    df["tier"] = np.select(conditions, choices, default="DEVELOP")

    # Override: high decline risk = RELEASE
    if "decline_risk_20pct" in df.columns:
        df.loc[df["decline_risk_20pct"] > 0.6, "tier"] = "RELEASE"

    log.info("  Tier distribution:")
    log.info("\n%s", df["tier"].value_counts().to_string())

    return df


def generate_summary(frontier: pd.DataFrame, all_players: pd.DataFrame) -> str:
    """Generate a text summary of the Pareto analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("PARETO FRONTIER ANALYSIS: Profitability vs. Sporting Risk")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total players analyzed: {len(all_players)}")
    lines.append(f"Pareto-optimal players: {len(frontier)}")
    lines.append("")
    lines.append("--- PARETO-OPTIMAL PLAYERS (Best Risk-Adjusted Value) ---")
    lines.append("")

    for _, row in frontier.iterrows():
        lines.append(
            f"  {row.get('short_name', '?'):20s} | "
            f"{row.get('team_name', '?'):20s} | "
            f"Age: {row.get('age_at_season_end', 0):5.1f} | "
            f"Value: {row.get('market_value_eur', 0):>12,.0f} EUR | "
            f"Profit: {row.get('profitability_score', 0):5.1f} | "
            f"Risk: {row.get('sporting_risk_score', 0):5.1f} | "
            f"Tier: {row.get('tier', '?')}"
        )

    lines.append("")
    lines.append("--- RECOMMENDATIONS ---")
    lines.append("")

    sell = all_players[all_players["tier"] == "SELL"]
    hold = all_players[all_players["tier"] == "HOLD"]
    develop = all_players[all_players["tier"] == "DEVELOP"]
    release = all_players[all_players["tier"] == "RELEASE"]

    lines.append(f"  SELL     ({len(sell)} players): Top risk-adjusted value, maximize sale window")
    for _, row in sell.head(10).iterrows():
        lines.append(f"    - {row.get('short_name', '?'):20s} (Profit: {row.get('profitability_score', 0):.1f}, Risk: {row.get('sporting_risk_score', 0):.1f})")

    lines.append(f"  HOLD     ({len(hold)} players): Valuable but higher risk, retain for sporting value")
    for _, row in hold.head(5).iterrows():
        lines.append(f"    - {row.get('short_name', '?'):20s} (Profit: {row.get('profitability_score', 0):.1f}, Risk: {row.get('sporting_risk_score', 0):.1f})")

    lines.append(f"  DEVELOP  ({len(develop)} players): Off-Pareto but promising, invest in development")
    for _, row in develop.head(5).iterrows():
        lines.append(f"    - {row.get('short_name', '?'):20s} (Profit: {row.get('profitability_score', 0):.1f}, Risk: {row.get('sporting_risk_score', 0):.1f})")

    lines.append(f"  RELEASE  ({len(release)} players): High decline risk, limited return potential")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def try_plot_pareto(all_players: pd.DataFrame, frontier: pd.DataFrame, fig_path: Path):
    """Attempt to create a Pareto scatter plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 9))

        # Color by tier
        tier_colors = {
            "SELL": "#2ecc71",
            "HOLD": "#f39c12",
            "DEVELOP": "#3498db",
            "RELEASE": "#e74c3c",
        }

        for tier, color in tier_colors.items():
            mask = all_players["tier"] == tier
            subset = all_players[mask]
            ax.scatter(
                subset["sporting_risk_score"],
                subset["profitability_score"],
                c=color,
                label=f"{tier} ({len(subset)})",
                alpha=0.6,
                s=40,
                edgecolors="white",
                linewidths=0.5,
            )

        # Highlight Pareto frontier
        frontier_sorted = frontier.sort_values("sporting_risk_score")
        ax.plot(
            frontier_sorted["sporting_risk_score"],
            frontier_sorted["profitability_score"],
            "k--",
            alpha=0.8,
            linewidth=2,
            label=f"Pareto Frontier ({len(frontier)})",
        )

        # Label top Pareto players
        for _, row in frontier.iterrows():
            ax.annotate(
                row.get("short_name", ""),
                (row["sporting_risk_score"], row["profitability_score"]),
                fontsize=7,
                alpha=0.8,
                ha="left",
                va="bottom",
            )

        ax.set_xlabel("Sporting Risk Score (lower = better)", fontsize=12)
        ax.set_ylabel("Profitability Score (higher = better)", fontsize=12)
        ax.set_title("Pareto Frontier: Profitability vs. Sporting Risk\nReal Madrid Player Sales Optimization", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved Pareto plot: %s", fig_path)

    except ImportError:
        log.info("  matplotlib not available, skipping plot")
    except Exception as e:
        log.warning("  Plot failed: %s", e)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    features, profiles, decline_pred = load_data()
    log.info("Loaded features: %s, profiles: %s", features.shape, profiles.shape)

    all_players, frontier = run_pareto_optimization(features, profiles, decline_pred)
    all_players = assign_tiers(all_players)

    all_players.to_csv(OUTPUT_DIR / "pareto_analysis.csv", index=False)
    frontier.to_csv(OUTPUT_DIR / "pareto_frontier.csv", index=False)
    log.info("Saved pareto_analysis.csv (%d rows)", len(all_players))
    log.info("Saved pareto_frontier.csv (%d rows)", len(frontier))

    summary = generate_summary(frontier, all_players)
    log.info("\n%s", summary)

    summary_path = OUTPUT_DIR / "pareto_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    log.info("Saved summary: %s", summary_path)

    try_plot_pareto(all_players, frontier, FIG_DIR / "pareto_frontier.png")

    return all_players, frontier


if __name__ == "__main__":
    main()
