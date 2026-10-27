"""
Phase 3 -- Financial & La Fabrica integration into the sales-optimisation data.

SSAC: financial value of player departures + recognition of Real Madrid
academy (La Fabrica) graduates.

DESIGN (per user direction -- no fuzzy row-level matching in the main
pipeline; fuzzy name-matching cannot disambiguate Spanish compound names and
would attach fees to the wrong players):

  Prong 1 -- Main features pipeline (row-level, CURATED ALLOWLIST only):
      is_la_fabrica_academy    1 for VERIFIED La Fabrica graduates active in
                               the 2022-25 dataset (curated player_id list);
                               else 0. No guessed links.
      purchase_fee_m           most recent 'in' transfer fee (EuroM)
      last_sale_fee_m          most recent 'out' fee (EuroM)
      market_value_m           dossier market value (EuroM, fallback features)
      accounting_profit_realized_m  last_sale_fee - purchase_fee (EuroM)
      projected_capital_gain_m      market_value - purchase_fee (EuroM)
      book_value_base_m        purchase fee (accounting cost proxy)
      mega_profit_potential    1 if projected_capital_gain_m >= 100
      is_profit_max            1 if accounting_profit_realized_m >= 20

  Prong 2 -- Macro Academy ROI reference (STANDALONE aggregate, no join):
      data/processed/la_fabrica_aggregate.csv
      Historical 2006-2026 La Fabrica: total sales volume, mean/median fee,
      income distribution, buy-back retained-rights value. This informs the
      academy-ROI narrative without corrupting per-player stats.

Financial joins (club_transfers / team_full_dossier) are namespaced by
Transfermarkt player_id, which does NOT match the internal features player_id,
and their names are full/compound while features uses FIFA-style names. We
therefore join the *financial* tables ONLY via a conservative exact-normalised
name + surname-unique match restricted to players actually present in features
(as a crosswalk), and we report honest coverage. The crosswalk mapping is a
best-effort; fee coverage is inherently low because these tables cover only a
subset of the roster and name formats differ.
"""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(r"C:\football")
DATA = ROOT / "data"
FEATURES = DATA / "processed" / "features.csv"
OUT = DATA / "processed" / "features_with_financials.csv"
AGGREGATE = DATA / "processed" / "la_fabrica_aggregate.csv"

SD_DB = DATA / "SD-DB(PLAYERS).csv"
CASES_XLSX = DATA / "real_madrid_la_fabrica_deep_analysis_expanded_2006_2026.xlsx"

# ---------------------------------------------------------------------------
# Prong 1 -- Curated La Fabrica graduate allowlist (VERIFIED, hand-checked
# against the 2022-25 Real Madrid squad in features.csv). These are RM
# academy / Castilla products present in the dataset.
# ---------------------------------------------------------------------------
LA_FABRICA_GRADUATES = {
    3304,    # Nacho Fernandez           (RM youth -> first team)
    3327,    # Joselu                   (RM Castilla)
    4498,    # Lucas Vazquez            (RM youth -> first team)
    4501,    # Dani Carvajal            (RM youth -> first team)
    5167,    # Mariano Diaz             (RM Castilla)
    326523,  # Dani Ceballos            (RM youth, re-signed)
    404397,  # Brahim Diaz              (RM youth, re-signed)
    422517,  # Fran Garcia              (RM Castilla, re-signed)
    536150,  # Sergio Arribas          (RM Castilla)
    601298,  # Lorenzo Aguado          (RM Castilla)
    626229,  # Mario Martin            (RM Castilla)
    626252,  # Gonzalo Garcia          (RM Castilla)
    626253,  # Nico Paz                (RM academy)
    770356,  # Daniel Yanez            (RM Castilla)
    780739,  # Victor Munoz            (RM Castilla)
    787348,  # Raul Asencio            (RM academy/Castilla)
    788748,  # Alvaro Rodriguez        (RM Castilla)
    846784,  # Jacobo Ramon            (RM Castilla)
    859506,  # Fran Gonzalez           (RM Castilla)
    859519,  # Chema Andres            (RM Castilla)
}


def _norm(s):
    """Strip accents, lowercase, keep alphanumerics."""
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = "".join(c for c in unicodedata.normalize("NFD", str(s))
                if unicodedata.category(c) != "Mn")
    s = "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ")
    return " ".join(s.split())


def _parse_m(x) -> float | None:
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.replace("\u20ac", "").replace("\x80", "").replace(" ", "").strip()
        if x in ("", "-", "n/a", "N/A"):
            return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def build_financial_crosswalk(fv: pd.DataFrame) -> tuple:
    """
    Conservative crosswalk from the financial CSVs (TFM-namespaced) to the
    internal features player_id. Conditions: (a) the player must be present in
    features, (b) the exact normalised full name agrees, or (c) the last
    surname token is unique among features AND the first-name agrees.
    Returns (buy_fee_map, out_fee_map, dossier_map).
    """
    transfers = pd.read_csv(DATA / "club_transfers.csv")
    dossier = pd.read_csv(DATA / "team_full_dossier.csv")

    ply = fv[["player_id", "first_name", "last_name"]].drop_duplicates("player_id")
    ply = ply.assign(_fn=ply["first_name"].fillna("").astype(str).map(_norm),
                     _ln=ply["last_name"].fillna("").astype(str).map(_norm),
                     _full=ply.apply(lambda r: _norm(f"{r['first_name']} {r['last_name']}"),
                                     axis=1))
    by_surname = {}
    full_index = set(ply["_full"])
    for _, r in ply.iterrows():
        ln = r["_ln"]
        if ln:
            by_surname.setdefault(ln.split()[-1], []).append((r["player_id"], r["_fn"]))

    def resolve(name):
        n = _norm(name)
        if not n:
            return None
        if n in full_index:
            row = ply[ply["_full"] == n]
            return int(row.iloc[0]["player_id"]) if len(row) == 1 else None
        toks = n.split()
        pool = by_surname.get(toks[-1], [])
        if len(pool) != 1:
            return None
        pid, fn = pool[0]
        return int(pid)

    def join_map(source_df, id_col):
        df = source_df.drop_duplicates(id_col)
        return {resolve(nm): v for nm, v in zip(df["player_name"], df["price"])
                if resolve(nm) is not None}

    buy_fee = join_map(
        transfers[transfers["direction"].eq("in")]
        .sort_values("transfer_date").drop_duplicates("player_id", keep="last"),
        "player_id")
    out_fee = join_map(
        transfers[transfers["direction"].eq("out")]
        .sort_values("transfer_date").drop_duplicates("player_id", keep="last"),
        "player_id")
    dossier_map = {
        int(r["player_id"]): (r["market_value_at_checkpoint"],
                              r["predicted_value_1y"], r["predicted_value_2y"])
        for _, r in dossier.sort_values("as_of_date")
                 .drop_duplicates("player_id", keep="last").iterrows()
        if resolve(r["player_name"]) is not None
    }
    return buy_fee, out_fee, dossier_map


def attach_financials(fv: pd.DataFrame) -> pd.DataFrame:
    """Attach La Fabrica flags + financial columns (curated only)."""
    fv["is_la_fabrica_academy"] = fv["player_id"].isin(LA_FABRICA_GRADUATES).astype(int)

    buy_fee, out_fee, dossier_map = build_financial_crosswalk(fv)
    fv["purchase_fee_m"] = fv["player_id"].map(buy_fee)
    fv["last_sale_fee_m"] = fv["player_id"].map(out_fee)
    fv["market_value_at_checkpoint"] = fv["player_id"].map(
        lambda pid: dossier_map.get(pid, (None, None, None))[0])
    fv["predicted_value_1y"] = fv["player_id"].map(
        lambda pid: dossier_map.get(pid, (None, None, None))[1])
    fv["predicted_value_2y"] = fv["player_id"].map(
        lambda pid: dossier_map.get(pid, (None, None, None))[2])

    if "market_value_eur" in fv.columns:
        fv["market_value_m"] = fv["market_value_at_checkpoint"].fillna(
            fv["market_value_eur"] / 1e6)
    else:
        fv["market_value_m"] = fv["market_value_at_checkpoint"]

    purchase = fv["purchase_fee_m"].fillna(0.0)
    fv["book_value_base_m"] = purchase
    fv["accounting_profit_realized_m"] = (
        fv["last_sale_fee_m"].fillna(0.0) - purchase)
    fv["projected_capital_gain_m"] = (
        fv["market_value_m"].fillna(0.0) - purchase)
    fv["mega_profit_potential"] = (fv["projected_capital_gain_m"] >= 100).astype(int)
    fv["is_profit_max"] = (fv["accounting_profit_realized_m"] >= 20).astype(int)

    log.info("La Fabrica graduates flagged (curated): %d",
             int(fv["is_la_fabrica_academy"].sum()))
    log.info("Financial coverage -> purchase_fee: %d (%.1f%%)",
             fv["purchase_fee_m"].notna().sum(),
             100 * fv["purchase_fee_m"].notna().sum() / len(fv))
    log.info("Financial coverage -> last_sale_fee: %d (%.1f%%)",
             fv["last_sale_fee_m"].notna().sum(),
             100 * fv["last_sale_fee_m"].notna().sum() / len(fv))
    log.info("Financial coverage -> market value: %d (%.1f%%)",
             fv["market_value_m"].notna().sum(),
             100 * fv["market_value_m"].notna().sum() / len(fv))
    return fv


def build_aggregate() -> pd.DataFrame:
    """Prong 2 -- standalone La Fabrica 2006-2026 academy-ROI aggregate table."""
    cases = pd.read_excel(CASES_XLSX, sheet_name="Cases")
    fee = cases["Initial fee (€M)"].map(_parse_m)
    net = cases["Direct net income (€M)"].map(_parse_m)
    bb = cases["Buy-back cost (€M)"].map(_parse_m)
    sellon = pd.to_numeric(cases["Sell-on (%)"], errors="coerce")
    agg = {
        "n_cases": len(cases),
        "n_with_fee": fee.notna().sum(),
        "total_sale_fee_m": fee.sum(),
        "mean_fee_m": fee.mean(),
        "median_fee_m": fee.median(),
        "max_fee_m": fee.max(),
        "total_net_income_m": net.sum(),
        "mean_net_income_m": net.mean(),
        "median_net_income_m": net.median(),
        "n_with_buyback": int(cases["Buy-back"].eq("Yes").sum()) if "Buy-back" in cases else 0,
        "n_with_sell_on": int(sellon.notna().sum()),
        "total_sell_on_pct": sellon.sum(),
        "n_bought_back": int(cases["Bought back"].eq("Yes").sum()) if "Bought back" in cases else 0,
        "n_sale_with_buyback": int(cases["Operation type"].str.contains(
            "buy-back", case=False, na=False).sum()),
    }
    return pd.DataFrame([agg])


def main() -> None:
    fv = pd.read_csv(FEATURES)
    out = attach_financials(fv)
    out.to_csv(OUT, index=False)
    log.info("Wrote %s (%d rows)", OUT, len(out))

    agg = build_aggregate()
    agg.to_csv(AGGREGATE, index=False)
    log.info("Wrote %s", AGGREGATE)
    for k, v in agg.iloc[0].items():
        if v is not None:
            log.info("  LaFabrica %-22s = %s", k, v)


if __name__ == "__main__":
    main()
