"""
build_features.py
=================
Derive profitability and sporting-risk features from the master dataset.

Feature families:
  A. Profitability (sell-side value)
  B. Sporting Risk (performance volatility / decline)
  C. Composite scoring

Output: data/processed/features.csv
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
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "features.csv"


def load_master() -> pd.DataFrame:
    return pd.read_csv(INTERIM_DIR / "master_dataset.csv")


# ======================================================================
# A. PROFITABILITY FEATURES
# ======================================================================

def build_profitability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features capturing the financial/sell value of a player."""
    log.info("  Building profitability features...")

    # --- A1: Core financial ---
    df["log_market_value"] = np.log1p(df["market_value_eur"])

    # --- A2: Market value change (YoY) ---
    df = df.sort_values(["player_id", "season_id"])
    df["value_change_yoy"] = df.groupby("player_id")["market_value_eur"].diff()
    df["value_change_pct"] = df.groupby("player_id")["market_value_eur"].pct_change()

    # --- A3: Minutes utilization rate ---
    max_minutes_per_season = 90 * 38  # full LaLiga season
    df["minutes_utilization"] = (
        df["stat_total_minutes_played"] / max_minutes_per_season
    ).clip(0, 1)

    # --- A4: Goal contributions per 90 ---
    goals = df.get("stat_total_goals", 0)
    assists = df.get("stat_total_assists", 0)
    minutes = df["stat_total_minutes_played"].replace(0, np.nan)
    df["goal_contributions_p90"] = ((goals + assists) / minutes * 90).fillna(0)

    # --- A5: Non-penalty goal contributions per 90 ---
    penalties = df.get("stat_total_penalties", 0)
    df["np_goal_contributions_p90"] = (
        ((goals + assists - penalties) / minutes * 90).fillna(0)
    )

    # --- A6: Shots efficiency ---
    shots = df.get("stat_total_shots", 0)
    shots_on = df.get("stat_total_shots_on_target", 0)
    df["shot_conversion"] = (goals / shots.replace(0, np.nan)).fillna(0)
    df["shot_on_target_rate"] = (shots_on / shots.replace(0, np.nan)).fillna(0)

    # --- A7: Pass completion ---
    passes = df.get("stat_total_passes", 0)
    passes_ok = df.get("stat_total_successful_passes", 0)
    df["pass_completion"] = (passes_ok / passes.replace(0, np.nan)).fillna(0)

    # --- A8: Progressive actions p90 ---
    prog_passes = df.get("stat_total_progressive_passes", 0)
    prog_runs = df.get("stat_total_progressive_run", 0)
    df["progressive_actions_p90"] = (
        ((prog_passes + prog_runs) / minutes * 90).fillna(0)
    )

    # --- A9: Key passes p90 ---
    key_passes = df.get("stat_total_key_passes", 0)
    df["key_passes_p90"] = ((key_passes / minutes * 90).fillna(0))

    # --- A10: Duels won rate ---
    duels = df.get("stat_total_duels", 0)
    duels_won = df.get("stat_total_duels_won", 0)
    df["duels_won_rate"] = (duels_won / duels.replace(0, np.nan)).fillna(0)

    # --- A11: Defending actions p90 ---
    def_actions = df.get("stat_total_defensive_actions", 0)
    interceptions = df.get("stat_total_interceptions", 0)
    recoveries = df.get("stat_total_recoveries", 0)
    df["defensive_actions_p90"] = (
        ((def_actions + interceptions + recoveries) / minutes * 90).fillna(0)
    )

    # --- A12: Dribble success ---
    dribbles = df.get("stat_total_dribbles", 0)
    dribbles_ok = df.get("stat_total_successful_dribbles", 0)
    df["dribble_success"] = (dribbles_ok / dribbles.replace(0, np.nan)).fillna(0)

    # --- A13: Value per goal contribution (lower = better value) ---
    gc = goals + assists
    df["value_per_gc"] = (
        df["market_value_eur"] / gc.replace(0, np.nan)
    )

    # --- A14: Value per minute ---
    df["value_per_minute"] = (
        df["market_value_eur"] / df["stat_total_minutes_played"].replace(0, np.nan)
    )

    # --- A15: Team league rank as proxy for selling context ---
    df["team_league_rank_filled"] = df["team_league_rank"].fillna(10)

    log.info("    Profitability features built: OK")
    return df


# ======================================================================
# B. SPORTING RISK FEATURES
# ======================================================================

def build_sporting_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features capturing the risk of performance decline or inconsistency."""
    log.info("  Building sporting risk features...")

    # --- B1: Age-based risk ---
    df["age_risk"] = np.where(
        df["age_at_season_end"] < 24, 0.1,
        np.where(df["age_at_season_end"] < 28, 0.0,
        np.where(df["age_at_season_end"] < 31, 0.2,
        np.where(df["age_at_season_end"] < 34, 0.5,
        0.8)))
    )

    # --- B2: Minutes risk (low minutes = injury/selection risk) ---
    max_minutes = 90 * 38
    df["minutes_risk"] = 1 - (df["stat_total_minutes_played"] / max_minutes).clip(0, 1)

    # --- B3: Starts ratio (sub usage = squad player risk) ---
    df["starts_ratio"] = (
        df["matches_started"] / df["matches"].replace(0, np.nan)
    ).fillna(0)

    # --- B4: Card risk (discipline) ---
    yellow = df.get("stat_total_yellow_cards", 0)
    red = df.get("stat_total_red_cards", 0)
    df["card_risk"] = (yellow * 1 + red * 3) / df["stat_total_minutes_played"].replace(0, np.nan) * 90
    df["card_risk"] = df["card_risk"].fillna(0)

    # --- B5: Fouls suffered (injury/physical risk proxy) ---
    fouls_suffered = df.get("stat_total_fouls_suffered", 0)
    df["fouls_suffered_p90"] = (
        (fouls_suffered / df["stat_total_minutes_played"].replace(0, np.nan) * 90).fillna(0)
    )

    # --- B6: Win contribution (team wins with/without player) ---
    df["win_rate_with"] = (
        df["wins"] / df["matches"].replace(0, np.nan)
    ).fillna(0)

    # --- B7: Position-specific risk multipliers ---
    position_risk = {
        "gk": 0.3, "lcb": 0.3, "rcb": 0.3, "cb": 0.3,
        "lb": 0.4, "rb": 0.4, "lwb": 0.5, "rwb": 0.5,
        "lcmf": 0.5, "rcmf": 0.5, "dmf": 0.4, "cmf": 0.5,
        "lamf": 0.6, "ramf": 0.6, "lw": 0.7, "rw": 0.7,
        "lwf": 0.7, "rwf": 0.7, "cf": 0.8, "ss": 0.8,
    }
    df["position_risk"] = df["primary_position"].map(position_risk).fillna(0.5)

    # --- B8: Age-position interaction (forwards decline faster) ---
    is_attacker = df["primary_position"].isin(["cf", "ss", "lw", "rw", "lwf", "rwf"])
    df["age_position_risk"] = np.where(
        is_attacker & (df["age_at_season_end"] > 30), 0.3, 0.0
    )

    # --- B9: Cross-season performance change ---
    df = df.sort_values(["player_id", "season_id"])
    perf_cols = ["stat_total_goals", "stat_total_assists", "stat_total_minutes_played"]
    for col in perf_cols:
        if col in df.columns:
            short_col = col.replace("stat_total_", "")
            df[f"{short_col}_change_yoy"] = df.groupby("player_id")[col].diff()

    # --- B10: Performance consistency (rolling std proxy) ---
    key_metric = df.get("goal_contributions_p90",
                         (df.get("stat_total_goals", 0) + df.get("stat_total_assists", 0))
                         / df["stat_total_minutes_played"].replace(0, np.nan) * 90)
    df["performance_metric"] = key_metric.fillna(0)
    df["performance_std"] = df.groupby("player_id")["performance_metric"].transform(
        lambda x: x.rolling(window=2, min_periods=1).std()
    )

    log.info("    Sporting risk features built: OK")
    return df


# ======================================================================
# C. COMPOSITE SCORING
# ======================================================================

def build_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Combine profitability and risk into normalized composite scores."""
    log.info("  Building composite scores...")

    # --- C1: Profitability Score (0-100, higher = more valuable to sell) ---
    profit_components = []
    profit_weights = []

    # Market value (log scale, normalized)
    if "log_market_value" in df.columns:
        mv = df["log_market_value"].fillna(df["log_market_value"].median())
        mv_norm = (mv - mv.min()) / (mv.max() - mv.min()) * 100
        profit_components.append(mv_norm)
        profit_weights.append(0.35)

    # Goal contributions p90
    if "goal_contributions_p90" in df.columns:
        gc = df["goal_contributions_p90"].fillna(0)
        gc_max = gc.quantile(0.95) if gc.max() > 0 else 1
        gc_norm = (gc / gc_max).clip(0, 1) * 100
        profit_components.append(gc_norm)
        profit_weights.append(0.25)

    # Minutes utilization
    if "minutes_utilization" in df.columns:
        mu = df["minutes_utilization"].fillna(0) * 100
        profit_components.append(mu)
        profit_weights.append(0.15)

    # Value change YoY (positive = growing asset)
    if "value_change_pct" in df.columns:
        vc = df["value_change_pct"].fillna(0).clip(-2, 5)
        vc_norm = ((vc + 2) / 7) * 100
        profit_components.append(vc_norm)
        profit_weights.append(0.15)

    # Pass completion (stability indicator)
    if "pass_completion" in df.columns:
        pc = df["pass_completion"].fillna(0) * 100
        profit_components.append(pc)
        profit_weights.append(0.10)

    if profit_components:
        total_weight = sum(profit_weights)
        df["profitability_score"] = sum(
            c * w / total_weight for c, w in zip(profit_components, profit_weights)
        ).round(2)
    else:
        df["profitability_score"] = 0.0

    # --- C2: Sporting Risk Score (0-100, higher = more risk) ---
    risk_components = []
    risk_weights = []

    # Age risk
    if "age_risk" in df.columns:
        risk_components.append(df["age_risk"] * 100)
        risk_weights.append(0.20)

    # Minutes risk
    if "minutes_risk" in df.columns:
        risk_components.append(df["minutes_risk"] * 100)
        risk_weights.append(0.20)

    # Card risk (normalized)
    if "card_risk" in df.columns:
        cr = df["card_risk"].clip(0, 3)
        cr_norm = (cr / 3) * 100
        risk_components.append(cr_norm)
        risk_weights.append(0.10)

    # Position risk
    if "position_risk" in df.columns:
        risk_components.append(df["position_risk"] * 100)
        risk_weights.append(0.15)

    # Age-position interaction
    if "age_position_risk" in df.columns:
        risk_components.append(df["age_position_risk"] * 100)
        risk_weights.append(0.15)

    # Performance decline (negative YoY = risk)
    if "goals_change_yoy" in df.columns:
        decline = df["goals_change_yoy"].fillna(0)
        decline_risk = np.where(decline < 0, (-decline).clip(0, 20) / 20 * 100, 0)
        risk_components.append(pd.Series(decline_risk, index=df.index))
        risk_weights.append(0.20)

    if risk_components:
        total_weight = sum(risk_weights)
        df["sporting_risk_score"] = sum(
            c * w / total_weight for c, w in zip(risk_components, risk_weights)
        ).round(2)
    else:
        df["sporting_risk_score"] = 0.0

    log.info("    Composite scores built: OK")
    return df


# ======================================================================
# MAIN
# ======================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading master dataset...")
    df = load_master()
    log.info("  Shape: %s", df.shape)

    df = build_profitability_features(df)
    df = build_sporting_risk_features(df)
    df = build_composite_scores(df)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    log.info("=" * 60)
    log.info("OUTPUT: %s", OUTPUT_FILE)
    log.info("Shape: %s", df.shape)
    log.info("Columns: %d", len(df.columns))
    log.info("=" * 60)

    # Quick summary
    log.info("\n--- Profitability Score distribution ---")
    log.info(df["profitability_score"].describe().to_string())
    log.info("\n--- Sporting Risk Score distribution ---")
    log.info(df["sporting_risk_score"].describe().to_string())

    # Top players by composite
    top = df.nlargest(10, "profitability_score")[
        ["short_name", "team_name", "season_name", "market_value_eur",
         "profitability_score", "sporting_risk_score"]
    ]
    log.info("\n--- Top 10 by profitability ---\n%s", top.to_string(index=False))

    return df


if __name__ == "__main__":
    main()
