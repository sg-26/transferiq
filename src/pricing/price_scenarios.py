# -*- coding: utf-8 -*-
"""
price_scenarios.py - Stage 14: add predicted fees + net financial impact.

Reads the committed scenario artifacts (scenarios.csv / scenario_players.csv,
UNTOUCHED - the flat 0.95 legacy view stays valid) and the extended pricing
master, then emits:

  data/processed/scenarios_priced.csv
      scenarios.csv + predicted_sale_fee, wage_savings, net_financial_impact,
      sale_revenue_legacy_0_95 (for the old-vs-new dashboard toggle)

  data/processed/scenario_pricing_per_player.csv
      per-scenario, per-player breakdown (fee, wage obligation avoided)
      for the drill-down view.

Definitions (documented in handoff):
  predicted_sale_fee        sum of per-player model fee (ridge+bucket blend)
  wage_savings              sum of annual_wage_eur * contract_years_remaining
                            (cash the club stops paying once the player leaves)
  net_financial_impact      predicted_sale_fee + wage_savings (total cash impact,
                            wages are a future-cost deduction the club avoids)
Both 'predicted' and 'legacy flat 0.95 (raise_value)' views are carried so
the dashboard can toggle and never silently replaces the RM/20M headline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(r"C:\football")
PROC = ROOT / "data" / "processed"
SCEN = PROC / "scenarios.csv"
SCEN_PLAYERS = PROC / "scenario_players.csv"
PRICING = ROOT / "data" / "interim" / "master_pricing.csv"
OUT = PROC / "scenarios_priced.csv"
OUT_PLAYER = PROC / "scenario_pricing_per_player.csv"


def main():
    scenarios = pd.read_csv(SCEN)
    players = pd.read_csv(SCEN_PLAYERS)
    pricing = pd.read_csv(PRICING)

    player_fin = pricing[["player_id", "season_id", "predicted_sale_fee",
                          "annual_wage_eur", "contract_years_remaining"]].copy()
    player_fin["wage_savings"] = (player_fin["annual_wage_eur"] *
                                  player_fin["contract_years_remaining"])
    player_fin = player_fin.rename(columns={
        "predicted_sale_fee": "player_predicted_fee",
        "annual_wage_eur": "player_annual_wage_eur"})

    pl = players.merge(
        player_fin[["player_id", "season_id", "player_predicted_fee",
                    "player_annual_wage_eur", "wage_savings"]],
        on=["player_id"], how="left").sort_values(
        ["scenario_id"]).reset_index(drop=True)
    # De-duplicate player rows that exist in multiple seasons of the pricing
    # master (scenario_players has no season column; all scenarios are RM 24/25).
    pl = pl.groupby(["scenario_id", "player_id", "short_name", "position_group",
                     "market_value"], as_index=False).agg(
        player_predicted_fee=("player_predicted_fee", "max"),
        player_annual_wage_eur=("player_annual_wage_eur", "max"),
        wage_savings=("wage_savings", "max"))
    pl["player_predicted_fee"] = pl["player_predicted_fee"].fillna(0.0)
    pl["wage_savings"] = pl["wage_savings"].fillna(0.0)

    per = pl[["scenario_id", "player_id", "short_name", "position_group",
              "market_value", "player_predicted_fee", "player_annual_wage_eur",
              "wage_savings"]].sort_values(["scenario_id", "player_predicted_fee"],
                                           ascending=[True, False])
    per.to_csv(OUT_PLAYER, index=False)

    g = pl.groupby("scenario_id").agg(
        predicted_sale_fee=("player_predicted_fee", "sum"),
        wage_savings=("wage_savings", "sum"))
    out = scenarios.merge(g, on="scenario_id", how="left")
    out["predicted_sale_fee"] = out["predicted_sale_fee"].fillna(0.0)
    out["wage_savings"] = out["wage_savings"].fillna(0.0)
    out["net_financial_impact"] = (out["predicted_sale_fee"] + out["wage_savings"])
    out["sale_revenue_legacy_0_95"] = out["raise_value"]
    out["predicted_vs_legacy_pct"] = np.where(
        out["sale_revenue_legacy_0_95"] > 0,
        100 * out["net_financial_impact"] / out["sale_revenue_legacy_0_95"], np.nan)

    out.to_csv(OUT, index=False)
    log.info("Wrote %s (%d rows) and %s (%d rows)",
             OUT, len(out), OUT_PLAYER, len(per))
    log.info("  median predicted fee per scenario: %.0f EUR | median wag+contract saved: %.0f EUR",
             out["predicted_sale_fee"].median(), out["wage_savings"].median())
    log.info("  RM 20M headline (S0): legacy=%.1fM predicted=%.1fM net=%.1fM",
             out.iloc[0]["sale_revenue_legacy_0_95"] / 1e6,
             out.iloc[0]["predicted_sale_fee"] / 1e6,
             out.iloc[0]["net_financial_impact"] / 1e6)


if __name__ == "__main__":
    main()