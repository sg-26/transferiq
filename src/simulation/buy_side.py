# -*- coding: utf-8 -*-
"""
buy_side.py - Stage 15: sell+buy (net) scenario engine.

For every feasible RM 24/25 sale scenario (the 3000 committed ones) this finds
affordable signings from the OTHER LaLiga clubs in the same season and scores
the NET outcome of "sell {A,B} + buy {C}" through the SAME shared scorer as the
batch path (single source of truth).

A scenario is "sell + buy": sell players that raise target capital, then spend
part of the realistic proceeds (sale_price_model predicted fees) on the best
affordable signing for the most-depleted position group.

Outputs:
  data/processed/scenario_signings.csv      chosen signing per scenario (if any)
  data/processed/scenario_pareto_net.csv    net frontier: net capital vs net xP
  data/processed/signing_candidates.csv     affordable candidate pool (dashboard)
"""
from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\football")
SRC_DIR = PROJECT_ROOT / "src"
for _p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from models.club_sporting_model import ClubSportingModel  # noqa: E402  (pickle pre-import)
from optimization import scenario_pareto as sp  # noqa: E402
from reporting import scenario_scoring as ss  # noqa: E402
from simulation import squad_counterfactual as scf  # noqa: E402

ROOT = Path(r"C:\football")
TEAM_ID = 675
SEASON_ID = 189974

DEFAULT_GROUP_PRIORITY = ["GK", "DF", "MF", "FW"]
BUY_MULTIPLIER = 1.0        # conservative: assume paying current market value
MAX_CANDIDATES_PER_SCENARIO = 3
MAX_SIGNINGS_TOTAL = 9000   # full coverage: 3000 scenarios x 3 candidate evals


def load_inputs():
    sq = scf.load_squad_data()
    team_name = pd.read_parquet(
        ROOT / "data" / "laliga_2022_2025" / "teams.parquet").set_index("team_id")["name"]
    price = pd.read_csv(ROOT / "data" / "interim" / "master_pricing.csv")
    scen_price = pd.read_csv(ROOT / "data" / "processed" / "scenarios_priced.csv")
    scen_players = pd.read_csv(ROOT / "data" / "processed" / "scenario_players.csv")
    scen = pd.read_csv(ROOT / "data" / "processed" / "scenarios.csv")
    model = sp.load_model()
    with open(ROOT / "models" / "sale_price_model.pkl", "rb") as fh:
        sm = pickle.load(fh)
    return sq, team_name, price, scen_price, scen_players, scen, model, sm


def build_candidate_pool(sq: pd.DataFrame, price: pd.DataFrame,
                         team_name: pd.Series) -> pd.DataFrame:
    """All non-RM LaLiga players active in 2024/25, with purchase estimate."""
    pool = sq[(sq["season_id"] == SEASON_ID) & (sq["team_id"] != TEAM_ID)].copy()
    pool = pool[pool["market_value_eur"].notna() & (pool["market_value_eur"] > 0)]
    fin = price[price["season_id"] == SEASON_ID][
        ["player_id", "predicted_sale_fee", "annual_wage_eur",
         "contract_years_remaining"]].drop_duplicates("player_id")
    pool = pool.merge(fin, on="player_id", how="left")
    pool["purchase_fee_est"] = pool["market_value_eur"] * BUY_MULTIPLIER
    pool["annual_wage_est"] = pool["annual_wage_eur"].fillna(
        pool["market_value_eur"] * 0.045)
    pool["wage_savings_never_incurred"] = (
        pool["annual_wage_est"] * pool["contract_years_remaining"].fillna(3.0))
    pool["team_name"] = pool["team_id"].map(team_name)
    pool["stat_gc"] = (pool.get("stat_total_goals", pd.Series(0.0, index=pool.index))
                       .fillna(0.0) + pool.get("stat_total_assists",
                                               pd.Series(0.0, index=pool.index)).fillna(0.0))
    pool["quality_per_m"] = np.where(
        pool["purchase_fee_est"] > 0,
        pool["stat_gc"] / (pool["purchase_fee_est"] / 1e6), 0.0)
    return pool.reset_index(drop=True)


def _state_with_bought(post_sell: pd.DataFrame, candidate: pd.Series) -> dict:
    row = candidate.to_dict()
    return scf.squad_state(pd.concat([post_sell, pd.DataFrame([row])],
                                     ignore_index=True))


def _score_net(sql_df: pd.DataFrame, base_state: dict,
               sold_ids: set, post_sell: pd.DataFrame,
               candidate: pd.Series | None, model) -> tuple:
    """Return (proceeds, purchase, net_capital, post_sell_delta, net_delta)."""
    if candidate is None:
        post_df = post_sell
        purchase = 0.0
    else:
        post_df = pd.concat([post_sell, pd.DataFrame([candidate.to_dict()])],
                            ignore_index=True)
        purchase = float(candidate["purchase_fee_est"])
    post_state = scf.squad_state(post_df)
    row = ss.build_scenario_row(base_state, post_state,
                                {"scenario_id": 0, "raise_value": 0.0,
                                 "total_sold_value": 0.0, "raise_excess": 0.0})
    scored = ss.score_scenarios_fast(model, pd.DataFrame([row]))
    return float(purchase), float(scored.iloc[0]["predicted_points_delta"])


def main():
    t0 = time.time()
    sq, team_name, price, scen_price, scen_players, scen, model, sm = load_inputs()
    engine = scf.SaleScenarioEngine(sq)
    squad = engine.get_squad(TEAM_ID, SEASON_ID)
    base_state = scf.squad_state(squad)
    pool = build_candidate_pool(sq, price, team_name)
    log.info("Candidate pool (non-RM LaLiga 24/25): %d players", len(pool))

    sold_map = scen_players.groupby("scenario_id")["player_id"].apply(set).to_dict()
    need_map = scen_players.groupby("scenario_id").agg(
        n_sold=("player_id", "size")).to_dict()["n_sold"]
    proceeds_map = scen_price.set_index("scenario_id")["predicted_sale_fee"].to_dict()

    # Baseline per-group depth counts for need detection
    base_counts = {g: int((squad["position_group"] == g).sum())
                   for g in DEFAULT_GROUP_PRIORITY}
    pool_by_group = {g: pool[pool["position_group"] == g].copy()
                     for g in DEFAULT_GROUP_PRIORITY}

    rows = []
    attempted = 0
    # Process scenarios ranked by predicted proceeds so the cap hits the most
    # interesting (cash-rich) deals first.
    for sid in sorted(range(len(scen)), key=lambda s: proceeds_map.get(s, 0),
                      reverse=True):
        ids = sold_map[sid]
        sold_df = squad[squad["player_id"].isin(ids)]
        post_sell = squad[~squad["player_id"].isin(ids)]
        sell_delta = _score_net(None, base_state, ids, post_sell, None, model)[1]

        budgets = proceeds_map.get(sid, 0.0)
        purchase = 0.0
        best_delta = sell_delta
        best_cand = None

        if attempted < MAX_SIGNINGS_TOTAL:
            # Position need: (1) real depth violation (below min) forces that
            # group; else (2) the group whose EXPECTED GOAL OUTPUT was most
            # damaged by the sale (attack-driven xP model recovers points there;
            # buying a GK/DF is good football but does not move predicted xP).
            rem = {g: base_counts[g] - int((sold_df["position_group"] == g).sum())
                   for g in DEFAULT_GROUP_PRIORITY}
            viol = [g for g in DEFAULT_GROUP_PRIORITY
                    if rem[g] < scf.DEFAULT_MIN_DEPTH[g]]
            if viol:
                need_group = viol[0]
            else:
                sold_gc = sold_df.groupby("position_group")["expected_gc"].sum()
                need_group = sold_gc.idxmax() if len(sold_gc) else "MF"
            cands = pool_by_group.get(need_group, pd.DataFrame())
            if cands.empty:
                need_group = DEFAULT_GROUP_PRIORITY[min(
                    base_counts, key=lambda g: base_counts[g])]
                cands = pool_by_group.get(need_group, pd.DataFrame())
            cands = cands[cands["purchase_fee_est"] <= budgets]
            cands = cands.sort_values(["stat_gc", "expected_gc"],
                                      ascending=False).head(
                MAX_CANDIDATES_PER_SCENARIO)
            for _, c in cands.iterrows():
                attempted += 1
                pur, net_delta = _score_net(None, base_state, ids, post_sell, c, model)
                if net_delta > best_delta:  # buying must improve on sell-only
                    best_delta, purchase, best_cand = net_delta, pur, c
        if attempted >= MAX_SIGNINGS_TOTAL and best_cand is None:
            pass  # no more buy attempts; keep sell-only net for remaining sids

        rows.append({
            "scenario_id": sid,
            "n_players_sold": int(need_map[sid]),
            "predicted_sale_fee": budgets,
            "sell_only_points_delta": sell_delta,
            "purchase_fee_est": purchase,
            "net_capital_eur": budgets - purchase,
            "net_points_delta": best_delta,
            "xP_recovery_points": best_delta - sell_delta,
            "signing_player_id": int(best_cand["player_id"]) if best_cand is not None else None,
            "signing_name": best_cand.get("short_name") if best_cand is not None else None,
            "signing_position": best_cand["position_group"] if best_cand is not None else None,
            "signing_team": best_cand.get("team_name") if best_cand is not None else None,
            "signing_age": float(best_cand["age"]) if best_cand is not None and not pd.isna(best_cand.get("age", np.nan)) else None,
            "signing_expected_gc": float(best_cand["expected_gc"]) if best_cand is not None else None,
        })

    sign_df = pd.DataFrame(rows)
    sign_df["signed"] = sign_df["signing_player_id"].notna()
    sign_df["net_capital_eur_m"] = sign_df["net_capital_eur"] / 1e6
    sign_df["sporting_cost_points"] = -sign_df["net_points_delta"]
    sign_df = sign_df.merge(
        scen[["scenario_id", "raise_value", "total_sold_value"]],
        on="scenario_id", how="left")
    sign_df.to_csv(PROC := ROOT / "data" / "processed" / "scenario_signings.csv",
                   index=False)

    par = sign_df.copy()
    par["pareto_optimal_net"] = sp.pareto_optimal(
        par, "net_capital_eur", "net_points_delta")
    par.to_csv(PROC2 := ROOT / "data" / "processed" / "scenario_pareto_net.csv",
               index=False)

    pool_out = pool[["player_id", "short_name", "team_name", "position_group",
                     "age", "market_value_eur", "expected_gc",
                     "decline_risk_20pct", "purchase_fee_est",
                     "annual_wage_est", "wage_savings_never_incurred",
                     "quality_per_m"]].sort_values("quality_per_m", ascending=False)
    pool_out.to_csv(ROOT / "data" / "processed" / "signing_candidates.csv", index=False)

    log.info("Wrote %s (%d rows, %d signed)", PROC, len(sign_df), int(sign_df["signed"].sum()))
    log.info("Wrote %s and signing_candidates.csv (%d pool rows)",
             PROC2, len(pool_out))
    fr = par[par["pareto_optimal_net"]].sort_values("net_points_delta")
    if not fr.empty:
        log.info("Net frontier: %d scenarios | extremes:", len(fr))
        log.info("  max capital : S%s sign=%s net=%.1fM cap=%.1fM xP=%.1f",
                 fr.iloc[-1]["scenario_id"], fr.iloc[-1]["signing_name"],
                 fr.iloc[-1]["net_capital_eur"] / 1e6,
                 fr.iloc[-1]["net_capital_eur"] / 1e6,
                 fr.iloc[-1]["net_points_delta"])
        log.info("  max points  : S%s sign=%s net=%.1fM cap=%.1fM xP=%.1f",
                 fr.iloc[0]["scenario_id"], fr.iloc[0]["signing_name"],
                 fr.iloc[0]["net_capital_eur"] / 1e6,
                 fr.iloc[0]["net_capital_eur"] / 1e6,
                 fr.iloc[0]["net_points_delta"])
    log.info("  attempted buy evals: %d | buy resolution rate: %.1f%%",
             attempted, 100 * sign_df["signed"].mean())
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()