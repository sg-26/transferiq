"""
squad_counterfactual.py
========================
Squad sale-scenario engine (the core decision unit).

Given a club + season and a target cash raise, generate candidate sale
scenarios (sets of players) and compute the counterfactual squad state
after those players depart.

Key concepts:
  - A "scenario" S_j = {P_1, ..., P_k} : a subset of the squad to sell.
  - Financial feasibility: sum(market_value) >= target_raise (with a
    margin factor since clubs cannot always fetch full market value).
  - Depth constraints: after departure, remaining squad must still hold
    minimum counts per position group.
  - Counterfactual squad features: recompute squad aggregates (value,
    goal concentration HHI, avg age, position depth, expected GC from
    retained players, aggregate decline risk) without the sold players.

IMPORTANT (sporting prediction caveat):
  There is NO valid club-level sporting model yet. The player decline model
  is player-level only and cannot project next-season league points/rank.
  This engine therefore computes COUNTERFACTUAL SQUAD FEATURES (state
  deltas) plus aggregate risk metrics from the retained squad. It does NOT
  fabricate a points/rank prediction. A club-level model can be plugged in
  later via `sporting_projector` (currently None).

Outputs:
  - data/processed/scenarios.csv            (one row per feasible scenario)
  - data/processed/scenario_players.csv     (player membership per scenario)
"""

import itertools
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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

POSITION_GROUP_MAP = {
    "gk": "GK",
    "lcb": "DF", "rcb": "DF", "cb": "DF", "lb": "DF", "rb": "DF", "lwb": "DF", "rwb": "DF",
    "lcb3": "DF", "rcb3": "DF", "lb5": "DF", "rb5": "DF", "cb3": "DF",
    "lcmf": "MF", "rcmf": "MF", "dmf": "MF", "cmf": "MF", "lamf": "MF", "ramf": "MF",
    "lcmf3": "MF", "rcmf3": "MF", "ldmf": "MF", "rdmf": "MF", "amf": "MF",
    "lw": "FW", "rw": "FW", "lwf": "FW", "rwf": "FW", "cf": "FW", "ss": "FW",
}

# Default minimum squad depth after sales (per position group)
DEFAULT_MIN_DEPTH = {"GK": 2, "DF": 6, "MF": 6, "FW": 4}


def map_position_group(pos: str) -> str:
    if pos in POSITION_GROUP_MAP:
        return POSITION_GROUP_MAP[pos]
    return POSITION_GROUP_MAP.get(pos.rstrip("0123456789"), "UNK")


def compute_hhi(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    total = arr.sum()
    if total == 0:
        return 0.0
    shares = arr / total
    return float(np.sum(shares ** 2))


def load_squad_data():
    """Load master + simulation results and build player-season records."""
    master = pd.read_csv(INTERIM_DIR / "master_dataset.csv")
    sim = pd.read_csv(PROCESSED_DIR / "simulation_results.csv")

    # Map season_name -> season_id
    season_map = {"2022/2023": 188124, "2023/2024": 189012, "2024/2025": 189974}
    sim["season_id"] = sim["season_name"].map(season_map)

    master = master.rename(columns={"age_at_season_end": "age"})

    # Join sim metrics onto master (per player_season)
    merge_cols = ["player_id", "season_id"]
    df = master.merge(
        sim[merge_cols + ["expected_gc", "gc_std", "decline_risk_20pct",
                          "decline_prob", "expected_value", "var_5pct",
                          "sharpe_ratio", "current_value"]],
        on=merge_cols,
        how="left",
        suffixes=("", "_sim"),
    )

    # Resolve any sim-vs-master value collisions
    if "market_value_eur" in df.columns and "current_value" in df.columns:
        df["market_value_eur"] = df["market_value_eur"].fillna(df["current_value"])

    df["position_group"] = df["primary_position"].apply(map_position_group)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce")
    df["expected_gc"] = pd.to_numeric(df["expected_gc"], errors="coerce").fillna(0)
    df["decline_risk_20pct"] = pd.to_numeric(
        df["decline_risk_20pct"], errors="coerce").fillna(0.5)
    df["decline_prob"] = pd.to_numeric(df["decline_prob"], errors="coerce").fillna(0.5)

    return df


def squad_state(members: pd.DataFrame) -> dict:
    """Compute squad-level aggregate state for a set of player-season rows."""
    n = len(members)
    if n == 0:
        return {
            "squad_size": 0, "total_market_value": np.nan,
            "avg_market_value": np.nan, "avg_age": np.nan, "u23": 0,
            "total_gc": 0.0, "total_expected_gc": 0.0,
            "avg_decline_risk": 0.0, "avg_decline_prob": 0.0,
            "goal_concentration_hhi": 0.0, "top_scorer_share": 0.0,
            "count_GK": 0, "count_DF": 0, "count_MF": 0, "count_FW": 0,
            "total_minutes": 0.0, "total_shots": 0.0, "total_assists": 0.0,
            "total_interceptions": 0.0, "squad_minutes_utilization": 0.0,
        }

    mv = members["market_value_eur"].dropna()
    gc = members[["stat_total_goals", "stat_total_assists"]] \
        .fillna(0).sum(axis=1)
    ages = members["age"].dropna()
    minutes = members.get("stat_total_minutes_played", pd.Series(0.0, index=members.index))
    minutes = pd.to_numeric(minutes, errors="coerce").fillna(0)

    def agg(col, key):
        if col in members.columns:
            return float(pd.to_numeric(members[col], errors="coerce").fillna(0).sum())
        return 0.0

    total_mins = float(minutes.sum())
    max_minutes = 90 * 38

    return {
        "squad_size": int(n),
        "total_market_value": float(mv.sum()) if len(mv) > 0 else np.nan,
        "avg_market_value": float(mv.mean()) if len(mv) > 0 else np.nan,
        "avg_age": float(ages.mean()) if len(ages) > 0 else np.nan,
        "u23": int((ages < 23).sum()) if len(ages) > 0 else 0,
        "total_gc": float(gc.sum()),
        "total_expected_gc": float(members["expected_gc"].sum()),
        "avg_decline_risk": float(members["decline_risk_20pct"].mean()),
        "avg_decline_prob": float(members["decline_prob"].mean()),
        "goal_concentration_hhi": compute_hhi(gc.values),
        "top_scorer_share": float(gc.max() / gc.sum()) if gc.sum() > 0 else 0.0,
        "count_GK": int((members["position_group"] == "GK").sum()),
        "count_DF": int((members["position_group"] == "DF").sum()),
        "count_MF": int((members["position_group"] == "MF").sum()),
        "count_FW": int((members["position_group"] == "FW").sum()),
        "total_minutes": total_mins,
        "total_shots": agg("stat_total_shots", "total_shots"),
        "total_assists": agg("stat_total_assists", "total_assists"),
        "total_interceptions": agg("stat_total_interceptions", "total_interceptions"),
        "squad_minutes_utilization": (
            total_mins / (max_minutes * n)) if n > 0 else 0.0,
    }


def to_model_features(state: dict) -> dict:
    """Convert a squad_state dict into the feature vector expected by the
    ClubSportingModel (see src/models/club_sporting_model.py extract_squad_features)."""
    return {
        "total_goal_contributions": state.get("total_gc", 0) or 0,
        "total_shots": state.get("total_shots", 0) or 0,
        "total_assists": state.get("total_assists", 0) or 0,
        "total_market_value": state.get("total_market_value", 0) or 0,
        "total_interceptions": state.get("total_interceptions", 0) or 0,
        "count_DF": state.get("count_DF", 0) or 0,
        "squad_minutes_utilization": state.get("squad_minutes_utilization", 0) or 0,
        "count_GK": state.get("count_GK", 0) or 0,
        "avg_age": state.get("avg_age", 26) or 26,
    }


def compute_feature_delta(base_state: dict, post_state: dict) -> dict:
    """Return dict of post-sale features plus absolute/relative deltas vs baseline."""
    delta = {}
    keys = ["squad_size", "total_market_value", "avg_market_value", "avg_age",
            "u23", "total_gc", "total_expected_gc", "avg_decline_risk",
            "avg_decline_prob", "goal_concentration_hhi", "top_scorer_share",
            "count_GK", "count_DF", "count_MF", "count_FW", "total_minutes"]
    for k in keys:
        b, p = base_state.get(k), post_state.get(k)
        delta[f"{k}_post"] = p

    delta["total_market_value_delta"] = (
        base_state["total_market_value"] - post_state["total_market_value"]
    ) if base_state["total_market_value"] is not None and post_state["total_market_value"] is not None \
      and not (pd.isna(base_state["total_market_value"]) or pd.isna(post_state["total_market_value"])) else np.nan

    if (pd.notna(base_state.get("total_expected_gc"))
            and base_state["total_expected_gc"] > 0
            and pd.notna(post_state.get("total_expected_gc"))):
        delta["expected_gc_retained_pct"] = post_state["total_expected_gc"] / base_state["total_expected_gc"]
    else:
        delta["expected_gc_retained_pct"] = np.nan

    # Depth constraint flags (remaining must be >= min depth)
    delta["depth_ok"] = all(
        post_state.get(f"count_{g}", 0) >= DEFAULT_MIN_DEPTH[g]
        for g in ["GK", "DF", "MF", "FW"]
    )

    return delta


class SaleScenarioEngine:
    def __init__(self, df: pd.DataFrame, min_depth: dict = None):
        self.df = df
        self.min_depth = min_depth or DEFAULT_MIN_DEPTH

    def get_squad(self, team_id: int, season_id: int) -> pd.DataFrame:
        return self.df[(self.df["team_id"] == team_id) & (self.df["season_id"] == season_id)].copy()

    def feasible_scenarios(self, squad: pd.DataFrame, target_raise: float,
                           max_players: int, sale_multiplier: float = 0.95,
                           max_scenarios: int = 5000) -> list:
        """
        Enumerate candidate sale scenarios up to `max_players`, keeping only
        those that (a) raise >= target_raise and (b) satisfy depth constraints.
        ```
        sale_multiplier < 1 because market value is a ceiling, not a booking fee.
        """
        results = []
        squad = squad.sort_values("market_value_eur", ascending=False).reset_index(drop=True)
        n = len(squad)
        max_k = min(max_players, n)

        # Pre-filter players we could even consider (value > 0)
        candidates = squad[squad["market_value_eur"].notna() & (squad["market_value_eur"] > 0)]

        # Baseline squad depth
        base_counts = {g: int((squad["position_group"] == g).sum()) for g in ["GK", "DF", "MF", "FW"]}

        for k in range(1, max_k + 1):
            # Prune: only subsets whose max achievable value (top k) can reach target
            if k > len(candidates):
                break
            topk_sum = candidates.head(k)["market_value_eur"].sum() * sale_multiplier
            if topk_sum < target_raise:
                # Even the most valuable k-player subset can't hit target, but the
                # top-k sum is MONOTONIC in k: a larger k may reach it. `continue`
                # (not `break`) so low-variance squads (e.g. Cadiz, where no single
                # player clears 20M but a 3-player set does) are not falsely pruned.
                continue

            for combo in itertools.combinations(candidates.index, k):
                sub = candidates.loc[list(combo)]
                raise_value = sub["market_value_eur"].sum() * sale_multiplier
                if raise_value < target_raise:
                    continue
                # Depth check on remaining squad
                rem_counts = {
                    g: base_counts[g] - int((sub["position_group"] == g).sum())
                    for g in ["GK", "DF", "MF", "FW"]
                }
                if all(rem_counts[g] >= self.min_depth[g] for g in ["GK", "DF", "MF", "FW"]):
                    results.append({
                        "scenario_id": len(results),
                        "player_ids": sub["player_id"].tolist(),
                        "raise_value": float(raise_value),
                        "total_value": float(sub["market_value_eur"].sum()),
                        "n_players": k,
                        "sub": sub,
                    })
                    if len(results) >= max_scenarios:
                        return results
        return results

    def evaluate_scenario(self, squad: pd.DataFrame, sub: pd.DataFrame,
                          base_state: dict) -> dict:
        """Compute counterfactual squad state and deltas for one scenario."""
        remaining = squad[~squad["player_id"].isin(sub["player_id"])].copy()
        post_state = squad_state(remaining)
        delta = compute_feature_delta(base_state, post_state)
        return post_state, delta


def build_scenarios(team_id: int, season_id: int, target_raise: float,
                    max_players: int, sale_multiplier: float = 0.95,
                    max_scenarios: int = 5000):
    df = load_squad_data()
    engine = SaleScenarioEngine(df)

    squad = engine.get_squad(team_id, season_id)
    if squad.empty:
        log.error("No squad found for team %s season %s", team_id, season_id)
        return pd.DataFrame(), pd.DataFrame()

    base_state = squad_state(squad)
    log.info("Baseline squad for team %s season %s:", team_id, season_id)
    log.info("  size=%s, total_value=%.0f, age=%.1f",
             base_state["squad_size"], base_state["total_market_value"],
             base_state["avg_age"])
    log.info("  depth GK=%d DF=%d MF=%d FW=%d",
             base_state["count_GK"], base_state["count_DF"],
             base_state["count_MF"], base_state["count_FW"])

    log.info("Enumerating feasible scenarios (target=%.0f, max_players=%d)...",
             target_raise, max_players)
    scenarios = engine.feasible_scenarios(squad, target_raise, max_players,
                                          sale_multiplier, max_scenarios)
    log.info("  %d feasible scenarios found", len(scenarios))

    scenario_rows = []
    player_rows = []
    # Baseline model features (shared across scenarios for this run)
    base_model_feats = to_model_features(base_state)
    for sc in scenarios:
        post_state, delta = engine.evaluate_scenario(squad, sc["sub"], base_state)
        post_model_feats = to_model_features(post_state)

        row = {
            "scenario_id": sc["scenario_id"],
            "team_id": team_id,
            "season_id": season_id,
            "n_players_sold": sc["n_players"],
            "raise_value": sc["raise_value"],
            "total_sold_value": sc["total_value"],
            "raise_excess": sc["raise_value"] - target_raise,
        }
        # Expose model-input capacity features for BASELINE and POST squad so
        # the ClubSportingModel can be applied downstream without recomputing.
        for k, v in base_model_feats.items():
            row[f"base_{k}"] = v
        for k, v in post_model_feats.items():
            row[f"post_{k}"] = v
        # Expose key counterfactual squad features
        for k in ["total_market_value_post", "avg_age_post", "total_gc_post",
                  "total_expected_gc_post", "avg_decline_risk_post",
                  "goal_concentration_hhi_post", "top_scorer_share_post",
                  "count_GK_post", "count_DF_post", "count_MF_post", "count_FW_post",
                  "expected_gc_retained_pct", "depth_ok"]:
            row[k] = delta.get(k, np.nan)
        scenario_rows.append(row)

        for _, p in sc["sub"].iterrows():
            player_rows.append({
                "scenario_id": sc["scenario_id"],
                "player_id": p["player_id"],
                "short_name": p.get("short_name"),
                "position_group": p["position_group"],
                "market_value": p["market_value_eur"],
                "age": p["age"],
                "expected_gc": p["expected_gc"],
                "decline_risk_20pct": p["decline_risk_20pct"],
            })

    scenarios_df = pd.DataFrame(scenario_rows)
    players_df = pd.DataFrame(player_rows)
    return scenarios_df, players_df


def run_demo():
    """Demonstrate: Real Madrid, 2024/25, raise target=20M, max 3 players."""
    team_id = 675   # Real Madrid
    season_id = 189974  # 2024/25
    target_raise = 20_000_000
    max_players = 3

    scenarios_df, players_df = build_scenarios(
        team_id, season_id, target_raise, max_players, max_scenarios=3000)

    out_dir = PROCESSED_DIR
    scenarios_df.to_csv(out_dir / "scenarios.csv", index=False)
    players_df.to_csv(out_dir / "scenario_players.csv", index=False)
    log.info("Saved scenarios.csv (%d rows) and scenario_players.csv (%d rows)",
             len(scenarios_df), len(players_df))

    log.info("\n--- Showcase: top scenarios by retained expected GC ---")
    if len(scenarios_df):
        top = scenarios_df.sort_values("expected_gc_retained_pct", ascending=False).head(5)
        log.info(top[["scenario_id", "n_players_sold", "raise_value",
                       "total_expected_gc_post", "expected_gc_retained_pct",
                       "avg_decline_risk_post", "depth_ok"]].to_string(index=False))


if __name__ == "__main__":
    run_demo()
