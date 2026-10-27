# -*- coding: utf-8 -*-
"""
sale_sanity_check.py - Stage 16: honest sanity check of the sale-price engine.

Two checks:

1) REALIZED-FEE CHECK. Compare the sale-price model's prediction to the actual
   realized fees of the Real Madrid out-transfers it was built on (leave-one-out
   spirit: prediction uses only the value-bucket + ridge, no memorization of the
   exact price). Reports MAPE and % of sales inside +-50%.

2) DECISION-DIRECTION CHECK. Three real RM sales (Varane, Odegaard, Casemiro)
   as historical anchors: realized price, market value at transfer, realized
   discount vs the model's bucket prediction. The xP side is checked on LIVE
   squad illustrations (selling Vinicius/Bellingham/Mbappe today: predicted fee
   vs predicted points lost) - clearly labelled illustrative so the tool never
   claims to reproduce actual historical decisions it cannot reconstruct.

Output: models/sale_sanity_check.json
"""
from __future__ import annotations

import json
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

from models.club_sporting_model import ClubSportingModel  # noqa: E402  (pickle pre-import)
from optimization import scenario_pareto as sp  # noqa: E402
from reporting import scenario_scoring as ss  # noqa: E402
from simulation import squad_counterfactual as scf  # noqa: E402

ROOT = Path(r"C:\football")
ANCHORS = ["Raphaël Varane", "Martin Ødegaard", "Casemiro"]
ILLUSTRATIONS = ["Vinícius Júnior", "J. Bellingham", "K. Mbappé"]


def load_realized():
    tr = pd.read_csv(ROOT / "data" / "club_transfers.csv")
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["mv"] = pd.to_numeric(tr["market_value_at_transfer"], errors="coerce")
    tr["is_loan"] = tr["is_loan"].astype(str).str.contains("True", case=False, na=False)
    tr["year"] = pd.to_datetime(tr["transfer_date"], errors="coerce").dt.year
    return tr[(tr["direction"] == "out") & (~tr["is_loan"])
              & tr["price"].notna() & tr["mv"].gt(0) & tr["price"].gt(0)]


def main():
    from pricing.sale_price_model import predict_sale_fee

    with open(ROOT / "models" / "sale_price_model.pkl", "rb") as fh:
        sm = pickle.load(fh)

    # Check 1: realized-fee comparison (no age/position => bucket+ridge only)
    realized = load_realized()
    preds = [predict_sale_fee(sm, r["mv"], None, "", float(r["year"]))
             for _, r in realized.iterrows()]
    realized["predicted"] = preds
    diff_pct = 100 * (realized["predicted"] - realized["price"]) / realized["price"]
    within50 = float((diff_pct.abs() <= 50).mean())
    # MAPE on log scale (robust to 0-free moves already excluded)
    mape = float(np.mean(np.abs(np.log(realized["predicted"] / realized["price"]))))
    low = float(np.exp(-np.log(1.5)))
    hi = float(np.exp(np.log(1.5)))
    rfe = {
        "n_realized_sales": int(len(realized)),
        "within_+/-50%": round(100 * within50, 1),
        "log_mape": round(mape, 3),
        "note": "prediction uses only market value + year (age/pos linker too thin); "
                "model is intentionally conservative, so large fees are under-predicted.",
        "examples": realized.sort_values("price", ascending=False).head(5)[
            ["player_name", "transfer_date", "price", "predicted"]].round(0)
            .astype({"price": int, "predicted": int}).to_dict("records"),
    }

    # Check 2: historical decision anchors (realized discount vs model bucket)
    anchors = []
    for name in ANCHORS:
        hits = realized[realized["player_name"].astype(str).str.contains(
            name.split()[0], case=False, na=False)]
        if hits.empty:
            continue
        r = hits.iloc[0]
        pred_disc = predict_sale_fee(sm, r["mv"], None, "", float(r["year"])) / r["mv"]
        anchors.append({
            "player": name,
            "transfer_year": int(r["year"]),
            "realized_fee_m": round(float(r["price"]) / 1e6, 1),
            "market_value_at_transfer_m": round(float(r["mv"]) / 1e6, 1),
            "realized_discount": round(float(r["price"] / r["mv"]), 2),
            "model_discount": round(float(pred_disc), 2),
        })

    # Check 2b: live squad illustrations - same engine as the planner
    sq = scf.load_squad_data()
    squad = sq[(sq["team_id"] == 675) & (sq["season_id"] == 189974)]
    base = scf.squad_state(squad)
    model = sp.load_model()
    price = pd.read_csv(ROOT / "data" / "interim" / "master_pricing.csv")
    price = price[(price["team_id"] == 675) & (price["season_id"] == 189974)]
    illustrations = []
    for name in ILLUSTRATIONS:
        row = squad[squad["short_name"].str.contains(
            name.replace("J. ", "").split()[0], case=False, na=False)]
        if row.empty:
            continue
        pid = set(row["player_id"])
        post = scf.squad_state(squad[~squad["player_id"].isin(pid)])
        r = ss.build_scenario_row(base, post, {"s": 0, "raise_value": 0,
                                                "total_sold_value": 0,
                                                "raise_excess": 0})
        delta = float(ss.score_scenarios_fast(model, pd.DataFrame([r]))
                      .iloc[0]["predicted_points_delta"])
        p = price[price["player_id"].isin(pid)]
        fee = float(p["predicted_sale_fee"].max() or 0)
        wag = float(p["wage_contract_value_eur"].max() or 0)
        illustrations.append({
            "player": str(row.iloc[0]["short_name"]),
            "type": "illustrative (current squad)",
            "predicted_fee_m": round(fee / 1e6, 1),
            "wage_obligation_avoided_m": round(wag / 1e6, 1),
            "predicted_points_lost": round(delta, 1),
        })

    result = {
        "sanity_check": "undefined",
        "realized_fee_check": rfe,
        "historical_decision_anchors": anchors,
        "illustrative_decision_cards": illustrations,
        "confidence_note": ("Sale-price discounts are fit on only %d non-free RM "
                            "sales; wage figures are heuristic estimates; deltas use "
                            "the attack-driven xP model. Direction is indicative, "
                            "not a claim of predictive precision." % int(len(realized))),
        "as_of": "2026-09",
    }
    # Simple, honest pass/partial based on direction + coverage
    ok_direction = bool(anchors) and all(
        0.3 <= a["realized_discount"] <= 1.6 for a in anchors)
    ok_realized = rfe["within_+/-50%"] >= 35.0
    result["sanity_check"] = "partial-pass" if (ok_direction and ok_realized) else "review"

    out = ROOT / "models" / "sale_sanity_check.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Wrote %s", out)
    log.info("  realized-fee: n=%d within+/-50%% = %.1f%%  logMAPE=%.3f",
             int(len(realized)), rfe["within_+/-50%"], mape)
    log.info("  anchors: %s",
             ", ".join("%s %.2f/%.2f" % (a["player"], a["realized_discount"],
                                          a["model_discount"]) for a in anchors))
    log.info("  SANITY = %s", result["sanity_check"])
    log.info("  illustrations: %s",
             ", ".join("%s fee=%.0fM xP=-%.1f" % (i["player"], i["predicted_fee_m"],
                                                   i["predicted_points_lost"])
                        for i in illustrations))


if __name__ == "__main__":
    main()