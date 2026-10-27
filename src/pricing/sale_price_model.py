# -*- coding: utf-8 -*-
"""
sale_price_model.py - Stage 12: learned sale-price + wage model.

Trains a small, transparent sale-price model on REAL Real Madrid transfer
fees (data/club_transfers.csv, the existing Transfermarkt pull) and bakes
`models/sale_price_model.pkl` for the pricing stages + dashboard.

Approach (honestly small data):
  * Training rows - non-loan OUT transfers with a recorded fee and a market
    value at transfer. Names are namespaced differently from the internal
    dataset, and most sold players are absent from the 2022-25 parquet, so
    age/position are NOT learnable from realized fees. n=18 non-free sales.
  * Target     - ln(discount), discount = fee / market_value_at_transfer.
  * Model      - ridge on [ln(market value), transfer year] -> calibrates how
    far below market value RM actually RECEIVES (historically ~0.5-0.9x).
    Prediction is the BLEND of the ridge fit and a transparent per-value-bucket
    empirical table, clipped to (0.35, 1.45). Flat 0.95 remains the fallback.
  * Age/position adjust the predicted fee as a DOCUMENTED HEURISTIC (not
    trained): older players sell at deeper discounts; GKs/DFs at a small
    premium to MV, FWs at a slight discount (transfer-market convention).
  * Wages - no wage column exists in any source, so annual wage is a labeled
    heuristic ~4.5% of market value, age/position-adjusted (est).
  * Contract - contract_years_remaining defaults to 3.0 (input in dashboard).
  * Buy multiplier - median fee/market-value on the 17 non-loan RM purchases.

Outputs:
  * models/sale_price_model.pkl  (model dict + fitted sales + meta)
  * data/processed/realized_sale_fees.csv  (transparent training table)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(r"C:\football")
DATA = ROOT / "data"
MODELS = ROOT / "models"
TRANSFERS = DATA / "club_transfers.csv"
OUT_PKL = MODELS / "sale_price_model.pkl"
OUT_SALES = DATA / "processed" / "realized_sale_fees.csv"

DISCOUNT_CLIP = (0.35, 1.45)
FALLBACK_MULTIPLIER = 0.95
WAGE_MV_RATE = 0.045  # annual wage ~4.5% of market value (est heuristic)


def norm(s):
    """ASCII-normalised name token (same convention as financial crosswalk)."""
    import unicodedata
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = "".join(c for c in unicodedata.normalize("NFD", str(s))
                if unicodedata.category(c) != "Mn")
    s = "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ")
    return " ".join(s.split())


def load_realized_sales() -> pd.DataFrame:
    tr = pd.read_csv(TRANSFERS)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["mv"] = pd.to_numeric(tr["market_value_at_transfer"], errors="coerce")
    tr["is_loan"] = tr["is_loan"].astype(str).str.contains("True", case=False, na=False)
    tr["year"] = pd.to_datetime(tr["transfer_date"], errors="coerce").dt.year

    out = tr[(tr["direction"] == "out") & (~tr["is_loan"]) & tr["price"].notna()].copy()
    out = out[out["mv"].notna() & (out["mv"] > 0)]
    out["discount"] = out["price"] / out["mv"]
    out["ln_discount"] = np.log(np.clip(out["discount"], 1e-6, None))
    out["ln_mv"] = np.log(out["mv"])
    out["non_free"] = out["price"] > 0
    return out


def fit_model(sales: pd.DataFrame) -> dict:
    train = sales[sales["non_free"]].copy()
    # Winsorize pathological outliers (e.g. an academy sale booked at 25x stale
    # market value) so they cannot dominate the regression or the buckets.
    train["discount_c"] = train["discount"].clip(0.1, 3.0)
    train["ln_discount"] = np.log(train["discount_c"])

    X = np.column_stack([train["ln_mv"], train["year"]]).astype(float)
    y = train["ln_discount"].to_numpy(dtype=float)
    mu_x = X.mean(axis=0)
    sd_x = X.std(axis=0) + 1e-9
    Xs = (X - mu_x) / sd_x
    w = np.linalg.solve(Xs.T @ Xs + 3.0 * np.eye(2), Xs.T @ y)
    b = y.mean() - w @ (X.mean(axis=0) / sd_x)

    def ridge_predict(ln_mv, year):
        x = np.array([ln_mv, year])
        return float(b + w @ ((x - mu_x) / sd_x))

    # Transparent per-value-bucket empirical table (winsorized medians)
    buckets = pd.cut(train["mv"],
                     bins=[0, 2e6, 10e6, 30e6, 60e6, np.inf],
                     labels=["<2M", "2-10M", "10-30M", "30-60M", ">60M"])
    table = (train.assign(bucket=buckets)
             .groupby("bucket", observed=True)["discount_c"].median().to_dict())
    for bkt in ["<2M", "2-10M", "10-30M", "30-60M", ">60M"]:
        table.setdefault(bkt, FALLBACK_MULTIPLIER)

    buy = pd.read_csv(TRANSFERS)
    buy["price"] = pd.to_numeric(buy["price"], errors="coerce")
    buy["mv"] = pd.to_numeric(buy["market_value_at_transfer"], errors="coerce")
    buy["is_loan"] = buy["is_loan"].astype(str).str.contains("True", case=False, na=False)
    in_tr = buy[(buy["direction"] == "in") & (~buy["is_loan"])
                & buy["price"].notna() & buy["mv"].gt(0)]
    buy_mult = float((in_tr["price"] / in_tr["mv"]).median()) if len(in_tr) else 1.0

    return {
        "version": 1,
        "kind": "ridge_discount_blend",
        "ridge_w": w.tolist(), "ridge_b": float(b),
        "ridge_mu_x": mu_x.tolist(), "ridge_sd_x": sd_x.tolist(),
        "bucket_table": {k: float(v) for k, v in table.items()},
        "discount_clip": list(DISCOUNT_CLIP),
        "fallback_multiplier": FALLBACK_MULTIPLIER,
        "n_fit": int(len(train)),
        "n_total_sales": int(len(sales)),
        "wage_mv_rate": WAGE_MV_RATE,
        "buy_multiplier": buy_mult,
        "buy_multiplier_hist": buy_mult,
        "buy_multiplier_default": 1.0,
        "train_note": (
            "Ridge on ln(discount) ~ ln(market value) + year over %d non-free RM "
            "out-transfers (club_transfers.csv). Age/position are documented "
            "heuristics, not learned (insufficient join coverage).") % int(len(train)),
        "fitted_sales": train[["player_name", "transfer_date", "price", "mv",
                               "discount", "age_heuristic_used"]]
        if "age_heuristic_used" in train.columns else
        train[["player_name", "transfer_date", "price", "mv", "discount"]],
    }


def _discount_for(model: dict, mv: float, year: float) -> float:
    """Blend ridge + per-value-bucket table, clipped."""
    try:
        w = np.asarray(model["ridge_w"])
        b = model["ridge_b"]
        mu_x = np.asarray(model["ridge_mu_x"])
        sd_x = np.asarray(model["ridge_sd_x"])
        x = (np.array([np.log(max(mv, 1e3)), year]) - mu_x) / sd_x
        ridge = float(b + w @ x)
    except Exception:
        ridge = 0.0
    bucket = (pd.cut([mv],
                     bins=[0, 2e6, 10e6, 30e6, 60e6, np.inf],
                     labels=["<2M", "2-10M", "10-30M", "30-60M", ">60M"])
              [0])
    tbl = float(model["bucket_table"].get(str(bucket), FALLBACK_MULTIPLIER))
    # The winsorized per-value-bucket median is the primary prediction
    # (historical realized realization curve). The ridge may only nudge it
    # within a +-15% band, so outlier academy sales cannot inflate stars.
    ratio = np.clip(np.exp(ridge) / max(tbl, 1e-6), 0.85, 1.15)
    blend = tbl * float(ratio)
    lo, hi = model["discount_clip"]
    return float(np.clip(blend, lo, hi))


def age_heuristic_factor(age) -> float:
    """Documented heuristic: peak 22-29, discount 30+, premium under 21.
    NOT learned (only 1 realized sale had joinable age)."""
    if age is None or age != age:
        return 1.0
    if age <= 21:
        return 1.05
    if age <= 29:
        return 1.0
    if age <= 33:
        return 0.88
    return 0.75


def position_heuristic_factor(position_group: str) -> float:
    """Documented transfer-market heuristic premium/discount by position."""
    g = str(position_group or "").upper()
    if g == "GK":
        return 1.02
    if g == "DF":
        return 1.0
    if g == "MF":
        return 0.97
    return 0.94  # FW: high nominal MV, often sells below it


def predict_sale_fee(model: dict, market_value_eur: float, age=None,
                     position_group: str = "", transfer_year: float = 2025) -> float:
    """Expected gross sale fee given current market value (+ heuristics)."""
    mv = float(market_value_eur or 0.0)
    if mv <= 0:
        return 0.0
    disc = _discount_for(model, mv, transfer_year)
    disc *= age_heuristic_factor(age)
    disc *= position_heuristic_factor(position_group)
    disc = float(np.clip(disc, model["discount_clip"][0], model["discount_clip"][1]))
    return mv * disc


def estimate_annual_wage_eur(market_value_eur: float, age=None,
                             position_group: str = "") -> float:
    """Estimated gross annual wage, ~4.5% of market value, age/position adjusted.

    HONESTY: no wage source exists in this repo. The derived column is an
    explicit estimate (dashboard labels it 'Est.') and is NOT a factual figure.
    """
    mv = float(market_value_eur or 0.0)
    if mv <= 0:
        return 0.0
    base = mv * WAGE_MV_RATE
    age_f = age_heuristic_factor(age)
    pos_f = position_heuristic_factor(position_group)
    est = base * (pos_f if pos_f <= 1.0 else 1.0) * age_f
    if age is not None and age != age and age >= 34:
        est *= 0.85
    return float(np.clip(est, 150_000, 25_000_000))


def main():
    sales = load_realized_sales()
    if not sales.empty:
        sales.to_csv(OUT_SALES, index=False)
        log.info("Wrote %s (%d rows)", OUT_SALES, len(sales))

    model = fit_model(sales)
    MODELS.mkdir(exist_ok=True)
    with open(OUT_PKL, "wb") as fh:
        pickle.dump(model, fh)
    log.info("Wrote %s", OUT_PKL)
    log.info("  n_fit=%d non-free sales; bucket table: %s",
             model["n_fit"], {k: round(v, 2) for k, v in model["bucket_table"].items()})
    log.info("  buy multiplier (hist median) = %.3f", model["buy_multiplier"])
    log.info("  ridge coefs (std-scale): ln_mv=%.3f, year=%.3f",
             model["ridge_w"][0], model["ridge_w"][1])
    for mv, age, pos in [(80_000_000, 26, "FW"), (5_000_000, 23, "MF"), (30_000_000, 31, "DF")]:
        log.info("  predict: mv=%.0f age=%s pos=%s -> fee=%.0f wage(est)=%.0f",
                 mv, age, pos, predict_sale_fee(model, mv, age, pos),
                 estimate_annual_wage_eur(mv, age, pos))


if __name__ == "__main__":
    main()