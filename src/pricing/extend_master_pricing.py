# -*- coding: utf-8 -*-
"""
extend_master_pricing.py - Stage 13: add sale-pricing + wage/contract columns.

Reads data/interim/master_dataset.csv (UNTOUCHED) and produces the extended
data/interim/master_pricing.csv consumed by scenario pricing, buy-side, and
the dashboard. Extends the existing dataset rather than rebuilding stage 3.

New columns (documented honestly):
  annual_wage_eur         ESTIMATED gross annual wage (~4.5% of market value,
                          age/position adjusted) - no wage source exists.
  contract_years_remaining  default 3.0 (user-adjustable in dashboard)
  predicted_sale_fee      sale_price_model fee (ridge+bucket blend x heuristics)
  sale_discount           predicted fee / market value
  last_realized_fee_eur   real fee of this player's most recent RM sale, if any
                          (via the stage-11 financial crosswalk -> honest anchor)
"""
from __future__ import annotations

import logging
import pickle
import sys
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

ROOT = Path(r"C:\football")
MASTER = ROOT / "data" / "interim" / "master_dataset.csv"
FIN = ROOT / "data" / "processed" / "features_with_financials.csv"
OUT = ROOT / "data" / "interim" / "master_pricing.csv"
PKL = ROOT / "models" / "sale_price_model.pkl"

SEASON_YEAR = {188124: 2023, 189012: 2024, 189974: 2025}
DEFAULT_CONTRACT_YEARS = 3.0


def map_position_group(pos):
    from simulation import squad_counterfactual as scf
    return scf.map_position_group(pos)


def load_sale_model():
    with open(PKL, "rb") as fh:
        return pickle.load(fh)


def main():
    from pricing.sale_price_model import estimate_annual_wage_eur, predict_sale_fee

    df = pd.read_csv(MASTER)
    model = load_sale_model()

    df["position_group"] = df["primary_position"].apply(map_position_group)
    age = pd.to_numeric(df["age_at_season_end"], errors="coerce")
    year = df["season_id"].map(SEASON_YEAR).fillna(2025)
    mv = pd.to_numeric(df["market_value_eur"], errors="coerce").fillna(0.0)

    df["predicted_sale_fee"] = [
        predict_sale_fee(model, float(m), a, g, int(y))
        for m, a, g, y in zip(mv, age, df["position_group"], year)]
    df["sale_discount"] = np.where(
        mv > 0, df["predicted_sale_fee"] / mv.replace(0, np.nan), np.nan)
    df["annual_wage_eur"] = [
        estimate_annual_wage_eur(float(m), a, g)
        for m, a, g in zip(mv, age, df["position_group"])]
    df["contract_years_remaining"] = DEFAULT_CONTRACT_YEARS
    df["wage_contract_value_eur"] = (
        df["annual_wage_eur"] * df["contract_years_remaining"])

    # Honest realized-fee anchors from the stage-11 crosswalk (Transfermarkt
    # namespace -> internal player_id), restricted to covered players.
    fin = pd.read_csv(FIN)[["player_id", "season_id", "purchase_fee_m",
                            "last_sale_fee_m", "is_la_fabrica_academy"]]
    df = df.merge(fin, on=["player_id", "season_id"], how="left")
    df["last_realized_fee_eur"] = df["last_sale_fee_m"] * 1e6
    df["purchase_fee_eur"] = df["purchase_fee_m"] * 1e6

    df.to_csv(OUT, index=False)
    log.info("Wrote %s (%d rows)", OUT, len(df))
    log.info("  predicted_sale_fee coverage: %.1f%%", 100 * df["predicted_sale_fee"].gt(0).mean())
    log.info("  annual_wage_eur (est) coverage: %.1f%% (heuristic, NOT factual)",
             100 * df["annual_wage_eur"].gt(0).mean())
    log.info("  accounted profit base coverage: last_realized_fee %.1f%% | purchase_fee %.1f%% | lafabrica %.1f%%",
             100 * df["last_realized_fee_eur"].notna().mean(),
             100 * df["purchase_fee_eur"].notna().mean(),
             100 * df["is_la_fabrica_academy"].notna().mean())
    rm = df[(df["team_id"] == 675) & (df["season_id"] == 189974)]
    log.info("  RM 24/25: n=%d, predicted sale value of squad=%.0f EUR (%.1fM)",
             len(rm), rm["predicted_sale_fee"].sum(), rm["predicted_sale_fee"].sum() / 1e6)


if __name__ == "__main__":
    main()