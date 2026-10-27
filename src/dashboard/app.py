# -*- coding: utf-8 -*-
"""
app.py - Interactive Streamlit decision product for the SSAC club system.

Stages 12-16 surface as product features:
  * Sale-value toggle      flat 0.95xMV (legacy headline) vs learned sale-price
                           model (realized RM fee curve + wages contract value).
  * Scenario Builder       build ANY custom sell (+ buy) scenario live, scored
                           with the SAME vectorized scorer as the batch path.
  * Budget targeting       "we need ~which capital at least sporting cost".
  * Compare                pin 2-3 scenarios side-by-side.
  * Depth check            live GK/DF/MF/FW validation, warning never crash.
  * Trust & method         honest sanity-check / limitations (Stage 16).

Scoring: single source of truth = reporting.scenario_scoring (verified 1:1
against ClubSportingModel.predict_scenario on the 3000 committed scenarios).
Core data + models load through streamlit caches so custom scoring is < 2 s.

Run:  python run_pipeline.py --dashboard     (or)   streamlit run src/dashboard/app.py
Run:  python src/dashboard/app.py --selfcheck       (headless chain + 3 custom cases)
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(r"C:\football")
SRC_DIR = PROJECT_ROOT / "src"
for _p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import streamlit as st

from models.club_sporting_model import ClubSportingModel  # noqa: E402,F401 (pickle pre-import)
from optimization import scenario_pareto as sp  # noqa: E402
from reporting import scenario_scoring as ss  # noqa: E402
from simulation import squad_counterfactual as scf  # noqa: E402

# Suppress harmless sklearn/Spacehenry warnings that fire during vectorized scoring
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.*")

SEASON_IDS = {188124: "2022/23", 189012: "2023/24", 189974: "2024/25"}
RM_TEAM_ID = 675
SEASON_ID_2024_25 = 189974
MAX_SCENARIOS = 3000
SALE_MULTIPLIER = 0.95
BUY_MULTIPLIER = 1.0
DEFAULT_MIN_DEPTH = scf.DEFAULT_MIN_DEPTH
GROUP_ORDER = ["GK", "DF", "MF", "FW"]


# ---------------------------------------------------------------------------
# Cached data loaders (models + big frames via cache_resource; small via cache_data)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_club_model():
    return sp.load_model()


@st.cache_resource(show_spinner=False)
def load_squad_frame():
    """Full club/season squad frame (all 3 seasons, all clubs).

    The builder + candidate pool derive from this frame, so removing the old
    2024/25 filter is what unlocks the Scenario Builder for every club-season.
    """
    return scf.load_squad_data()


@st.cache_data(show_spinner=False)
def load_pricing_master() -> pd.DataFrame:
    try:
        return pd.read_csv(PROJECT_ROOT / "data" / "interim" / "master_pricing.csv")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_candidate_pool() -> pd.DataFrame:
    """Rich sell+buy candidate pool (all non-RM LaLiga players, 24/25 season)."""
    try:
        return pd.read_csv(PROJECT_ROOT / "data" / "processed" / "signing_candidates.csv")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_scenarios_priced() -> pd.DataFrame:
    try:
        df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "scenarios_priced.csv")
        return df.set_index("scenario_id")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_scenario_signings() -> pd.DataFrame:
    try:
        df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "scenario_signings.csv")
        return df.set_index("scenario_id")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_sanity() -> dict:
    p = PROJECT_ROOT / "models" / "sale_sanity_check.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_club_catalog() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "club_season_dataset.csv")
    return df[["team_id", "team_name", "season_id"]].drop_duplicates().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_financials() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features_with_financials.csv")
    keep = [c for c in ["player_id", "season_id", "short_name", "market_value_eur",
                        "purchase_fee_m", "last_sale_fee_m", "is_la_fabrica_academy"]
            if c in df.columns]
    return df[keep].drop_duplicates().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_la_fabrica_aggregate() -> dict:
    p = PROJECT_ROOT / "data" / "processed" / "la_fabrica_aggregate.csv"
    if not p.exists():
        return {}
    row = pd.read_csv(p).iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


@st.cache_data(show_spinner=False)
def load_la_fabrica_cases() -> pd.DataFrame:
    p = PROJECT_ROOT / "data" / "real_madrid_la_fabrica_deep_analysis_expanded_2006_2026.xlsx"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(p, sheet_name="Cases")
        fee = next((c for c in df.columns if "Initial fee" in c), None)
        net = next((c for c in df.columns if "Direct net income" in c), None)
        if fee:
            df[fee] = pd.to_numeric(df[fee], errors="coerce")
        if net:
            df[net] = pd.to_numeric(df[net], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Core analysis (reuses the committed batch engine, scoring via shared module)
# ---------------------------------------------------------------------------
def _cap_depth_to_squad(min_depth: dict, base_counts: dict) -> dict:
    """Absolute depth floors can never exceed the baseline squad headcount.

    Structural rescue: Barcelona 24/25 has FW=3 (default floor FW>=4), so the
    default depth dogmatically forbids selling anything. Flat floored floors
    turn an impossible ask into a solvable one.
    """
    return {g: max(0, min(int(min_depth.get(g, DEFAULT_MIN_DEPTH[g])),
                         int(base_counts.get(g, 0))))
            for g in GROUP_ORDER}


def _max_achievable_raise(squad: pd.DataFrame, max_players: int,
                          min_depth: dict, sale_multiplier: float = 0.95):
    """Largest raise reachable under depth floors and the player limit.

    Returns (best_raise, best_combo). Used only when the strict target finds
    nothing, so the UI can say how much capital is actually achievable."""
    squad = squad.sort_values("market_value_eur", ascending=False).reset_index(drop=True)
    base_counts = {g: int((squad["position_group"] == g).sum()) for g in GROUP_ORDER}
    candidates = squad[squad["market_value_eur"].notna() & (squad["market_value_eur"] > 0)]
    best = 0.0
    best_combo = None
    for k in range(1, min(max_players, len(candidates)) + 1):
        for combo in itertools.combinations(candidates.index, k):
            sub = candidates.loc[list(combo)]
            raise_value = float(sub["market_value_eur"].sum() * sale_multiplier)
            if raise_value <= best:
                continue
            rem = {g: base_counts[g] - int((sub["position_group"] == g).sum())
                   for g in GROUP_ORDER}
            if all(rem[g] >= min_depth[g] for g in GROUP_ORDER):
                best, best_combo = raise_value, combo
    return best, best_combo


def _attach_learned_net(players_df: pd.DataFrame, pricing: pd.DataFrame,
                        season_id: int):
    """Per-scenario learned fee + wage totals live from master_pricing.

    Gives Real Madrid non-2024/25 seasons the same learned net-financial view
    the committed 2024/25 batch already has. Returns (fees, wages) joined on
    scenario_id, or (None, None) when no pricing is available."""
    if pricing.empty or players_df.empty:
        return None, None
    pm = pricing[(pricing["season_id"] == season_id) &
                 pricing["predicted_sale_fee"].notna()][
        ["player_id", "predicted_sale_fee", "annual_wage_eur",
         "contract_years_remaining"]].drop_duplicates("player_id").copy()
    if pm.empty:
        return None, None
    pm["_fee"] = pd.to_numeric(pm["predicted_sale_fee"], errors="coerce").fillna(0.0)
    pm["_wage"] = (pd.to_numeric(pm["annual_wage_eur"], errors="coerce").fillna(0.0)
                   * pd.to_numeric(pm["contract_years_remaining"], errors="coerce").fillna(0.0))
    m = players_df.merge(pm[["player_id", "_fee", "_wage"]], on="player_id", how="left")
    g = m.groupby("scenario_id")[["_fee", "_wage"]].sum()
    return g["_fee"], g["_wage"]


def compute_analysis(team_id: int, season_id: int, target_raise_eur: float,
                     max_players: int, min_depth: dict, allow_relaxation: bool = True):
    df = scf.load_squad_data()
    engine = scf.SaleScenarioEngine(df, min_depth=dict(min_depth))
    squad = engine.get_squad(team_id, season_id)
    if squad.empty:
        return None

    base_state = scf.squad_state(squad)
    base_counts = {g: int((squad["position_group"] == g).sum()) for g in GROUP_ORDER}
    floor_depth = _cap_depth_to_squad(dict(min_depth), base_counts)
    engine_use = scf.SaleScenarioEngine(df, min_depth=floor_depth)

    scenarios = engine_use.feasible_scenarios(
        squad, target_raise_eur, max_players,
        sale_multiplier=SALE_MULTIPLIER, max_scenarios=MAX_SCENARIOS)
    target_reached = len(scenarios) > 0
    achieved_max_m = 0.0
    depth_relaxed = floor_depth != dict(min_depth)

    if not target_reached and allow_relaxation and max_players >= 1:
        # Strict target found nothing -> surface the closest feasible set:
        # everything in a band down to 80% of the maximum achievable capital
        # under the same floors, so the user still gets options to look at.
        max_raise, _best = _max_achievable_raise(squad, max_players, floor_depth)
        achieved_max_m = float(max_raise) / 1e6
        if max_raise and max_raise > 0:
            relax_target = max(max_raise * 0.8, max_raise - 2_000_000)
            scenarios = engine_use.feasible_scenarios(
                squad, float(relax_target), max_players,
                sale_multiplier=SALE_MULTIPLIER, max_scenarios=MAX_SCENARIOS)

    scenario_rows, player_rows = [], []
    for sc in scenarios:
        post_state, delta = engine.evaluate_scenario(squad, sc["sub"], base_state)
        delta["depth_ok"] = all(
            post_state.get(f"count_{g}", 0) >= engine_use.min_depth[g]
            for g in GROUP_ORDER)
        row = ss.build_scenario_row(base_state, post_state, {
            "scenario_id": sc["scenario_id"],
            "team_id": team_id, "season_id": season_id,
            "n_players_sold": sc["n_players"],
            "raise_value": sc["raise_value"],
            "total_sold_value": sc["total_value"],
            "raise_excess": sc["raise_value"] - target_raise_eur,
        })
        for k in ss.COUNTERFACTUAL_COLS:
            row[k] = delta.get(k, np.nan)
        scenario_rows.append(row)

        for _, p in sc["sub"].iterrows():
            player_rows.append({
                "scenario_id": sc["scenario_id"],
                "player_id": p.get("player_id"),
                "short_name": p.get("short_name"),
                "position_group": p["position_group"],
                "market_value": p["market_value_eur"],
                "age": p["age"],
                "expected_gc": p["expected_gc"],
                "decline_risk_20pct": p["decline_risk_20pct"],
            })

    scenarios_df = pd.DataFrame(scenario_rows)
    players_df = pd.DataFrame(player_rows)
    if scenarios_df.empty:
        scenarios_df = pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in ss.scenario_schema_columns()})

    model = load_club_model()
    scored = ss.score_scenarios_fast(model, scenarios_df)
    scenarios_df = pd.concat([scenarios_df, scored], axis=1)
    scenarios_df["sporting_cost"] = -scenarios_df["predicted_points_delta"]

    # ---------- Capital basis ----------
    # The committed learned-net artifact (predicted_sale_fee / wage_savings /
    # net_financial_impact) is priced for the Real Madrid 2024/25 batch ONLY.
    # Merging it by bare scenario_id into any other club/season leaked RM fee
    # totals onto (e.g.) Barcelona scenarios with the same id -> the stale-RM
    # cross-contamination bug. Gate it to that one batch; everyone else gets
    # the flat market-value proxy (0.95 x MV) unless RM (live learned view).
    is_rm = int(team_id) == RM_TEAM_ID
    is_rm_batch = is_rm and int(season_id) == SEASON_ID_2024_25
    priced = load_scenarios_priced() if is_rm_batch else pd.DataFrame()
    if not priced.empty:
        scenarios_df = scenarios_df.merge(
            priced[["predicted_sale_fee", "wage_savings", "net_financial_impact",
                    "sale_revenue_legacy_0_95"]].reset_index(),
            on="scenario_id", how="left")
        scenarios_df["net_financial_impact"] = scenarios_df["net_financial_impact"].fillna(
            scenarios_df["raise_value"])
    elif is_rm:
        fees, wages = _attach_learned_net(players_df, load_pricing_master(), season_id)
        scenarios_df["predicted_sale_fee"] = np.nan
        scenarios_df["wage_savings"] = np.nan
        if fees is not None:
            scen_fees = scenarios_df["scenario_id"].map(fees).fillna(0.0)
            scen_wages = scenarios_df["scenario_id"].map(wages).fillna(0.0)
            scenarios_df["predicted_sale_fee"] = scen_fees
            scenarios_df["wage_savings"] = scen_wages
            scenarios_df["net_financial_impact"] = scen_fees + scen_wages
        else:
            scenarios_df["net_financial_impact"] = scenarios_df["raise_value"]
    else:
        scenarios_df["net_financial_impact"] = scenarios_df["raise_value"]
        scenarios_df["predicted_sale_fee"] = np.nan
        scenarios_df["wage_savings"] = np.nan
    scenarios_df["sale_revenue_legacy_0_95"] = scenarios_df["raise_value"]
    scenarios_df["revenue_eur_m"] = scenarios_df["raise_value"] / 1e6
    scenarios_df["net_eur_m"] = scenarios_df["net_financial_impact"].fillna(
        scenarios_df["raise_value"]) / 1e6
    scenarios_df["risk_change_pp"] = scenarios_df["predicted_relegation_risk_change"] * 100.0
    scenarios_df["pareto_optimal"] = sp.pareto_optimal(
        scenarios_df, "raise_value", "predicted_points_delta")
    meta = {
        "target_reached": target_reached,
        "achieved_max_m": achieved_max_m,
        "depth_relaxed": depth_relaxed,
        "floor_depth": dict(floor_depth),
        "is_rm": is_rm,
    }
    return scenarios_df, players_df, base_state, meta


compute_analysis_cached = st.cache_data(compute_analysis, show_spinner="Running scenario engine...")

# ---------------------------------------------------------------------------
# Custom scenario builder (sell + buy) - plain functions, live-scored
# ---------------------------------------------------------------------------
def squad_roster(sq: pd.DataFrame, team_id: int, season_id: int) -> pd.DataFrame:
    return sq[(sq["team_id"] == team_id) & (sq["season_id"] == season_id)].copy()


def candidate_roster(sq: pd.DataFrame, team_id: int, season_id: int,
                     pricing: pd.DataFrame) -> pd.DataFrame:
    """Rich buy candidate pool (all columns needed for squad_state scoring).

    Non-squad LaLiga players for the SELECTED season (club-agnostic; the
    committed signing_candidates.csv is a RM 24/25 export, this is live)."""
    pool = sq[(sq["season_id"] == season_id) & (sq["team_id"] != team_id)].copy()
    pool = pool[pool["market_value_eur"].notna() & (pool["market_value_eur"] > 0)]
    if not pricing.empty:
        fin = pricing[pricing["season_id"] == season_id][
            ["player_id", "predicted_sale_fee"]].drop_duplicates("player_id")
        pool = pool.merge(fin, on="player_id", how="left")
    pool["purchase_fee_est"] = pool["market_value_eur"] * BUY_MULTIPLIER
    pool["buy_label"] = (pool["short_name"].fillna("?") + "  [" +
                         pool["position_group"] + " | " +
                         pool["team_name"].astype(str).str.slice(0, 12) + "] " +
                         (pool["market_value_eur"] / 1e6).round(1).astype(str) + "M")
    return pool.reset_index(drop=True)


def score_sell_buy(sq: pd.DataFrame, team_id: int, season_id: int,
                   sold_ids: list, bought_ids: list, pool: pd.DataFrame,
                   base_state: dict, model, pricing: pd.DataFrame) -> dict:
    """Score an arbitrary sell(+buy) scenario through the shared scorer.

    Returns a dict with capital, model fee, net financial impact, purchase,
    points/relegation deltas, depth + names. Never raises on empty depth."""
    squad = squad_roster(sq, team_id, season_id)
    sold = squad[squad["player_id"].isin(sold_ids)]
    remaining = squad[~squad["player_id"].isin(sold_ids)]
    depth_ok = all(
        int((remaining["position_group"] == g).sum()) >= DEFAULT_MIN_DEPTH[g]
        for g in GROUP_ORDER)

    buys = pool[pool["player_id"].isin(bought_ids)] if len(bought_ids) else pd.DataFrame()
    post_df = remaining
    if not buys.empty:
        post_df = pd.concat([remaining, buys], ignore_index=True)
    post_state = scf.squad_state(post_df)

    row = ss.build_scenario_row(base_state, post_state, {
        "scenario_id": -1, "raise_value": 0.0, "total_sold_value": 0.0,
        "raise_excess": 0.0})
    scored = ss.score_scenarios_fast(model, pd.DataFrame([row]))
    r = scored.iloc[0]

    flat_raise = float(sold["market_value_eur"].sum() * SALE_MULTIPLIER)
    is_rm = int(team_id) == RM_TEAM_ID
    fee = flat_raise if not is_rm else 0.0
    wage = 0.0
    # Learned fee + wage contract view is trained on ~18 realized RM sales and
    # enriched into master_pricing for Real Madrid; other clubs fall back to
    # the flat market-value proxy (0.95 x MV) with zero wage surplus.
    if is_rm and not pricing.empty and not sold.empty:
        pm = pricing[(pricing["player_id"].isin(sold["player_id"])) &
                     (pricing["season_id"] == season_id)]
        if not pm.empty:
            fee = float(pm["predicted_sale_fee"].sum())
            wage = float((pm["annual_wage_eur"] * pm["contract_years_remaining"]).sum())
    purchase = float(buys["purchase_fee_est"].sum()) if not buys.empty else 0.0
    net_impact = fee + wage  # total cash impact incl. wages no longer paid
    return {
        "sell_ids": list(sold_ids), "buy_ids": list(bought_ids),
        "sold_names": [str(x) for x in sold["short_name"].tolist()],
        "buy_names": [str(x) for x in buys["short_name"].tolist()] if not buys.empty else [],
        "buy_teams": [str(x) for x in buys["team_name"].tolist()] if not buys.empty else [],
        "flat_raise": flat_raise, "model_fee": fee, "wage_savings": wage,
        "net_impact": net_impact, "purchase": purchase,
        "net_capital": fee - purchase,
        "points_baseline": float(r["predicted_points_baseline"]),
        "points_delta": float(r["predicted_points_delta"]),
        "relegation_change": float(r["predicted_relegation_risk_change"]),
        "depth_ok": depth_ok,
        "post_counts": {g: int((post_df["position_group"] == g).sum()) for g in GROUP_ORDER},
        "counts_sold": {g: int((sold["position_group"] == g).sum()) for g in GROUP_ORDER},
    }


def match_batch_signing(sold_ids: list, signings: pd.DataFrame,
                        players_by_sid: dict) -> tuple:
    """If the custom sell-set matches a committed scenario, return its batch
    recommended signing (name, team, position, purchase) as glue advice."""
    key = frozenset(int(x) for x in sold_ids)
    sid = next((s for s, ids in players_by_sid.items() if frozenset(ids) == key), None)
    if sid is None:
        return None
    r = signings.loc[sid] if sid in signings.index else None
    if r is None or pd.isna(r.get("signing_player_id")):
        return None
    return {"scenario_id": int(sid),
            "name": r["signing_name"], "team": r["signing_team"],
            "position": r["signing_position"],
            "purchase_m": float(r["purchase_fee_est"]) / 1e6,
            "recovery": float(r["xP_recovery_points"]) if pd.notna(r.get("xP_recovery_points")) else 0.0}


# ---------------------------------------------------------------------------
# Plotting / display helpers
# ---------------------------------------------------------------------------
def build_scatter(scenarios_df, overlay=None):
    import plotly.graph_objects as go
    x = -scenarios_df["predicted_points_delta"]
    y = scenarios_df["revenue_eur_m"]
    custom = scenarios_df[["scenario_id", "n_players_sold", "raise_value",
                           "predicted_points_delta",
                           "predicted_relegation_risk_change", "net_eur_m"]].values
    frontier = scenarios_df[scenarios_df["pareto_optimal"]].sort_values(
        "predicted_points_delta")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=-frontier["predicted_points_delta"], y=frontier["revenue_eur_m"],
        mode="lines", name="Pareto frontier (flat 0.95xMV)",
        line=dict(color="#888", width=2, dash="dot"),
        hovertemplate="Frontier<br>Points lost %{x:.1f}<br>Capital %{y:.1f}M<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="Scenarios",
        marker=dict(size=8, color="#5B8DB8", opacity=0.45,
                    line=dict(width=0.5, color="#33475B")),
        customdata=custom,
        hovertemplate=(
            "<b>Scenario %{customdata[0]}</b><br>"
            "Players sold: %{customdata[1]}<br>"
            "Capital raised: %{customdata[2]:,.0f} EUR (%{customdata[2]:,.1f}M)<br>"
            "Net impact (model): %{customdata[5]:,.1f}M<br>"
            "Points lost: %{customdata[3]:.1f}<br>"
            "Relegation risk change: %{customdata[4]:+.2%}<extra></extra>"),
    ))
    best = scenarios_df[scenarios_df["pareto_optimal"]]
    fig.add_trace(go.Scatter(
        x=-best["predicted_points_delta"], y=best["revenue_eur_m"],
        mode="markers", name="Pareto-optimal",
        marker=dict(size=15, color="#E3B341", symbol="star",
                    line=dict(width=1.5, color="#7A5A00")),
        customdata=best[["scenario_id", "n_players_sold", "raise_value",
                         "predicted_points_delta",
                         "predicted_relegation_risk_change"]].values,
        hovertemplate=(
            "<b>Pareto scenario %{customdata[0]}</b><br>"
            "Players sold: %{customdata[1]}<br>"
            "Capital raised: %{customdata[2]:,.0f} EUR (%{customdata[2]:,.1f}M)<br>"
            "Points lost: %{customdata[3]:.1f}<br>"
            "Relegation risk change: %{customdata[4]:+.2%}<extra></extra>"),
    ))
    if overlay:
        for lab, ox, oy in overlay:
            fig.add_trace(go.Scatter(
                x=[ox], y=[oy], mode="markers", name=lab,
                marker=dict(size=18, color="#19C37D", symbol="diamond",
                            line=dict(width=2, color="#0A6B45")),
                hovertemplate=f"<b>{lab}</b><br>Points lost {ox:.1f}<br>Capital {oy:.1f}M<extra></extra>",
            ))
    fig.update_layout(
        height=560,
        xaxis_title="Predicted points lost (smaller is better for sporting risk)",
        yaxis_title="Capital raised (EUR M, flat basis)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="closest",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def before_after_rows(base_state, scenario_row):
    rows = [
        ("Squad size", base_state["squad_size"], scenario_row.get("squad_size_post")),
        ("Market value (EUR)", base_state["total_market_value"],
         scenario_row.get("post_total_market_value")),
        ("Goal contributions (G+A)", base_state["total_gc"],
         scenario_row.get("post_total_goal_contributions")),
        ("Shots", base_state["total_shots"], scenario_row.get("post_total_shots")),
        ("Assists", base_state["total_assists"], scenario_row.get("post_total_assists")),
        ("Minutes utilization", base_state["squad_minutes_utilization"],
         scenario_row.get("post_squad_minutes_utilization")),
        ("Goal concentration (HHI)", base_state["goal_concentration_hhi"],
         scenario_row.get("goal_concentration_hhi_post")),
        ("Avg age", base_state["avg_age"], scenario_row.get("post_avg_age")),
        ("GK depth", base_state["count_GK"], scenario_row.get("count_GK_post")),
        ("DF depth", base_state["count_DF"], scenario_row.get("count_DF_post")),
        ("MF depth", base_state["count_MF"], scenario_row.get("count_MF_post")),
        ("FW depth", base_state["count_FW"], scenario_row.get("count_FW_post")),
        ("Expected GC retained (%)", 100.0,
         scenario_row.get("expected_gc_retained_pct", np.nan) * 100
         if scenario_row.get("expected_gc_retained_pct") is not None
         and not pd.isna(scenario_row.get("expected_gc_retained_pct")) else np.nan),
    ]
    out = []
    for label, b, a in rows:
        if pd.isna(b) or b is None:
            out.append({"Metric": label, "Before": "-", "After": "-", "Delta": "-"})
            continue
        b_disp = round(float(b), 3)
        a_disp = None if (a is None or pd.isna(a)) else round(float(a), 3)
        d_disp = None if a_disp is None else round(a_disp - b_disp, 3)
        out.append({"Metric": label, "Before": b_disp, "After": a_disp, "Delta": d_disp})
    return pd.DataFrame(out)


def format_scenario_label(row):
    tag = "  [Pareto]" if bool(row["pareto_optimal"]) else ""
    return (f"#{int(row['scenario_id'])} | sold {int(row['n_players_sold'])} | "
            f"raise {row['revenue_eur_m']:,.1f}M EUR | "
            f"{-float(row['predicted_points_delta']):.1f} pts lost{tag}")


def _agg_num(v):
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:,.2f}" if abs(f) < 100 else f"{f:,.0f}"


def render_la_fabrica(agg, cases, financials):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Academy cases tracked", _agg_num(agg.get("n_cases", agg.get("total_cases"))))
    c2.metric("Total sale fees (EUR M)",
              _agg_num(agg.get("total_sale_fees_m", agg.get("total_initial_fees_m"))))
    c3.metric("Direct net income (EUR M)",
              _agg_num(agg.get("direct_net_income_m", agg.get("net_income_m"))))
    c4.metric("Academy graduates in squad", _agg_num(agg.get("n_graduates_in_squad")))

    fee_col = next((c for c in cases.columns if "Initial fee" in c), None)
    if not cases.empty and fee_col:
        st.subheader("Top academy graduate sales (2006-2026)")
        top = cases.dropna(subset=[fee_col]).sort_values(fee_col, ascending=False).head(10)
        top["label"] = top["Player"] + " (" + top["Year left"].astype(str) + ")"
        st.bar_chart(top.set_index("label")[fee_col], horizontal=True)
        st.caption("Initial transfer fee in EUR M per academy graduate case.")

    graduates = financials[financials.get(
        "is_la_fabrica_academy", pd.Series(0, index=financials.index)) == 1]
    if not graduates.empty:
        st.subheader(f"La Fabrica academy graduates in current squads ({len(graduates)})")
        show = graduates[["short_name", "season_id", "market_value_eur",
                          "purchase_fee_m", "last_sale_fee_m"]]
        show = show.sort_values("market_value_eur", ascending=False).head(20)
        show["market_value_eur"] = show["market_value_eur"] / 1e6
        st.dataframe(show.rename(columns={
            "short_name": "Player", "season_id": "Season",
            "market_value_eur": "Market value (EUR M)",
            "purchase_fee_m": "Purchase fee (EUR M)",
            "last_sale_fee_m": "Last sale fee (EUR M)"}),
            hide_index=True, width="stretch",
            column_config={
                "Market value (EUR M)": st.column_config.NumberColumn(format="%.1f"),
                "Purchase fee (EUR M)": st.column_config.NumberColumn(format="%.1f"),
                "Last sale fee (EUR M)": st.column_config.NumberColumn(format="%.1f")})


def render_trust_tab():
    sanity = load_sanity()
    st.subheader("Trust & method")
    if sanity:
        rfe = sanity.get("realized_fee_check", {})
        anchors = sanity.get("historical_decision_anchors", [])
        st.markdown(
            f"**Sanity verdict: `{sanity.get('sanity_check', 'unknown')}`** - "
            "the sale-price model is checked against "
            f"**{rfe.get('n_realized_sales', '?')}** real Real Madrid transfers "
            f"(`{rfe.get('within_+/-50%', '?')}%` of fees within +/-50%).")
        if anchors:
            st.markdown("Realized transfer anchors (realised vs model discount):")
            st.dataframe(pd.DataFrame(anchors),
                         hide_index=True, width="stretch")
        st.caption(sanity.get("confidence_note", ""))
    st.markdown("""**Honest limitations**
- Sale price & wage figures are **management estimates**: the fee curve is fit
  on only ~18 non-free RM sales; wages are a `~4.5% of market value` heuristic.
- xP deltas are **attack-driven** by construction (defence model not used for
  deltas, R2=0.53) - so buying a GK/DF shows little points recovery.
- Signings come from the other 19 LaLiga clubs in the 2024/25 dataset only.
- No causal identification; no currency/inflation adjustment across seasons.
- Multi-window sequential squad simulation is a backlog / v2 idea.
""")
    st.markdown("""**Backlog / v2**
- multi-window sequential squad simulation (sell this season, buy next)
- richer transfer-fee data (transfermarkt full history) once available offline
- contract renegotiation inputs and two-year wage bill modelling
""")


# ---------------------------------------------------------------------------
# Pin / compare helpers
# ---------------------------------------------------------------------------
def pin_label(entry):
    if entry["kind"] == "custom":
        s = entry.get("sell_names", [])
        b = entry.get("buy_names", [])
        who = ", ".join(s) + (f"  +  {', '.join(b)}" if b else "")
        return f"Custom: {who}  ({entry.get('pts', 0.0):+.1f} pts, {entry.get('cap', 0.0):,.0f}M)"
    return f"#{entry.get('sid')}: sold {entry.get('n')} | {entry.get('cap', 0.0):,.1f}M | {entry.get('pts', 0.0):+.1f} pts"


def render_pinned_compare():
    pins = st.session_state.get("pins", [])
    if not pins:
        st.info("Pin scenarios from the Planner or the Custom Builder to compare here "
                "(up to 3).")
        return
    st.caption("Side-by-side comparison of pinned scenarios.")
    cols = st.columns(min(len(pins), 3))
    for col, ent in zip(cols, pins[:3]):
        with col:
            st.markdown(f"**{pin_label(ent)}**")
            if ent["kind"] == "custom":
                st.metric("Capital (model fee)", f"{ent['cap']:,.0f}M EUR")
                st.metric("Net impact", f"{ent.get('net', 0.0):,.0f}M EUR")
                st.metric("Purchase (buy)",
                          f"{ent.get('purchase', 0.0):,.0f}M EUR")
                st.metric("Points lost", f"{ent.get('pts', 0.0):+.1f}")
                st.metric("Relegation swing", f"{ent.get('rel', 0.0):+.2%}")
                st.markdown("**Sold:** " + ", ".join(ent.get("sell_names", [])))
                if ent.get("buy_names"):
                    st.markdown("**Buy:** " + ", ".join(
                        f"{n} ({t})" for n, t in zip(ent["buy_names"], ent.get("buy_teams", []))))
            else:
                st.metric("Capital", f"{ent['cap']:,.1f}M EUR")
                st.metric("Points lost", f"{ent.get('pts', 0.0):+.1f}")
                st.metric("Relegation swing", f"{ent.get('rel', 0.0):+.2%}")
            if st.button("Remove", key=f"rm_{ent.get('_uid')}", use_container_width=True):
                st.session_state["pins"] = [p for p in pins if p.get("_uid") != ent.get("_uid")]
                st.rerun()
    if len(pins) > 3:
        st.caption(f"{len(pins)} pinned - showing the first 3.")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="TransferIQ | Counterfactual Squad Decision Engine",
                       page_icon=":soccer:", layout="wide")

    sq = load_squad_frame()
    model = load_club_model()
    pricing = load_pricing_master()

    st.session_state.setdefault("pins", [])
    next_uid = int(st.session_state.get("pin_uid", 0))

    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Scenario controls")
        season_option = st.selectbox("Season", list(SEASON_IDS.values()), index=2)
        season_id = {v: k for k, v in SEASON_IDS.items()}[season_option]

        catalog = load_club_catalog()
        subset = catalog[catalog["season_id"] == season_id]
        names = sorted(subset["team_name"].unique())
        default_idx = names.index("Real Madrid") if "Real Madrid" in names else 0
        club_name = st.selectbox("Club", names, index=default_idx)
        team_id = int(subset[subset["team_name"] == club_name]["team_id"].iloc[0])

        # Adaptive financial target (no flat 20M assumption): bound the slider
        # by 5%-40% of the selected squad's total market value.
        sel_sq = sq[(sq["team_id"] == team_id) & (sq["season_id"] == season_id)]
        base_mv_m = float(sel_sq["market_value_eur"].sum()) / 1e6 if len(sel_sq) else 0.0
        lo = max(1, int(round(0.05 * base_mv_m)))
        hi = max(lo + 1, int(round(0.40 * base_mv_m)))
        step = 5 if hi > 40 else 1
        tgt_default = min(max(int(round(0.10 * base_mv_m)), lo), hi)
        target_raise_m = st.slider("Target revenue (EUR M)", lo, hi, tgt_default, step)
        max_players = st.slider("Max players sold", 1, 4, 3)
        st.caption("Minimum squad depth after sales")
        gk_min = st.slider("GK", 0, 6, 2)
        df_min = st.slider("DF", 0, 15, 6)
        mf_min = st.slider("MF", 0, 14, 6)
        fw_min = st.slider("FW", 0, 10, 4)
        capital_basis = st.radio(
            "Capital shown as",
            ["Flat 0.95xMV (legacy)", "Model fee + wages (net)"],
            help="Legacy = market value * 0.95 (the original RM/20M headline basis). "
                 "Net = learned sale price model plus contract/wage cash avoided.")
        is_rm = int(team_id) == RM_TEAM_ID
        if capital_basis.startswith("Model") and is_rm:
            capital_col = "net_eur_m"
        else:
            capital_col = "revenue_eur_m"
            if capital_basis.startswith("Model") and not is_rm:
                st.info(f"Advanced learned transfer fee and wage model is currently "
                       f"available for Real Madrid based on historical sales data. "
                       f"Displaying flat market value proxy for {club_name}.")
        st.markdown("---")
        sanity = load_sanity()
        if sanity.get("sanity_check") == "review":
            st.caption("Trust: sale-price model is directional, not precisely "
                       "calibrated vs 18 realized RM sales - treat fees as estimates.")
        else:
            st.caption("Trust: sale-price model sanity-checked on realized RM sales.")
        st.caption("This is a decision-support tool, not investment advice.")

    # Main heading -- neutral brand root, with the selected club appended once
    # the sidebar selectors have resolved (TransferIQ branding always leads).
    st.title(f"TransferIQ — {club_name}")
    st.caption("Counterfactual decision intelligence for player sales and "
               "squad reinvestment.")

    min_depth = {"GK": gk_min, "DF": df_min, "MF": mf_min, "FW": fw_min}

    # ------------------------------------------------------------------ tabs
    tab_planner, tab_builder, tab_compare, tab_lafab, tab_trust = st.tabs(
        ["Planner", "Scenario Builder", "Compare", "La Fabrica & Financials",
         "Trust & method"])

    # ---------------- Planner ----------------
    with tab_planner:
        result = compute_analysis_cached(int(team_id), int(season_id),
                                         float(target_raise_m) * 1e6, int(max_players),
                                         tuple(sorted(min_depth.items())))
        if result is None:
            st.error(f"No squad found for {club_name} in {season_option}.")
            st.stop()
        scenarios_df, players_df, base_state, meta = result

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Total market value", f"{base_state['total_market_value'] / 1e6:,.0f}M EUR")
        mc2.metric("Squad size", f"{base_state['squad_size']}")
        mc3.metric("Average age", f"{base_state['avg_age']:.1f}")
        mc4.metric("Goal concentration (HHI)", f"{base_state['goal_concentration_hhi']:.3f}")
        mc5.metric("Depth GK / DF / MF / FW",
                   f"{base_state['count_GK']}/{base_state['count_DF']}/"
                   f"{base_state['count_MF']}/{base_state['count_FW']}")

        if scenarios_df.empty:
            st.warning("No saleable players, or no sale combination can satisfy "
                       "the depth constraints for this club/season. Try lowering "
                       "the depth floors or adjusting the player limit.")
            st.stop()

        if not meta["target_reached"]:
            msg = (f"**No sale combination reached the {target_raise_m:,.0f}M EUR target** "
                   f"under the selected depth / player-limit constraints. Showing the "
                   f"closest feasible scenarios instead - maximum achievable capital "
                   f"is ~{meta['achieved_max_m']:,.0f}M EUR.")
            if meta.get("depth_relaxed"):
                msg += (" Depth floors were relaxed toward the squad's actual size "
                        "to keep the problem solvable.")
            st.warning(msg)

        # budget-target sub-mode
        budget_mode = st.toggle("Budget targeting: 'we need ~which capital with least "
                                "sporting cost'", value=False)
        if budget_mode:
            st.subheader(f"Scenarios within ~{target_raise_m}M of target "
                         f"(basis: {'model net' if capital_col == 'net_eur_m' else 'flat'})")
            cap = scenarios_df[capital_col]
            tol = max(target_raise_m * 0.25, 5.0)
            near = scenarios_df[(cap - target_raise_m).abs() <= tol].copy()
            near = near.assign(_cap=cap)
            near = near.sort_values("predicted_points_delta")  # least sporting cost first
            near = near.head(12)
            st.dataframe(near[[
                "scenario_id", "n_players_sold", "predicted_sale_fee", "wage_savings",
                "net_financial_impact", "predicted_points_delta",
                "predicted_relegation_risk_change", "pareto_optimal"]].rename(columns={
                    "scenario_id": "Scenario", "n_players_sold": "Players",
                    "predicted_sale_fee": "Model fee (EUR)", "wage_savings": "Wages+contract (EUR)",
                    "net_financial_impact": "Net impact (EUR)",
                    "predicted_points_delta": "xP delta",
                    "predicted_relegation_risk_change": "Rel risk change",
                    "pareto_optimal": "Pareto"}),
                hide_index=True, width="stretch",
                column_config={
                    "Model fee (EUR)": st.column_config.NumberColumn(format="%.0f"),
                    "Wages+contract (EUR)": st.column_config.NumberColumn(format="%.0f"),
                    "Net impact (EUR)": st.column_config.NumberColumn(format="%.0f"),
                    "xP delta": st.column_config.NumberColumn(format="%.1f"),
                    "Rel risk change": st.column_config.NumberColumn(format="%.4f"),
                    "Pareto": st.column_config.CheckboxColumn()})
            st.caption(f"Sorted by sporting cost (fewest points lost first). "
                       f"{len(near)} scenarios within {tol:.0f}M of {target_raise_m}M target.")
            st.stop()

        st.subheader("Pareto frontier - capital raised vs sporting risk")
        st.caption(
            f"{len(scenarios_df)} feasible scenarios for {club_name} {season_option}, "
            f"target {target_raise_m:,.0f}M EUR, up to {max_players} players. "
            f"{int(scenarios_df['pareto_optimal'].sum())} Pareto-optimal scenarios "
            "(gold stars). Frontier stars are fixed on the flat legacy axis. "
            + (f"Showing the closest feasible set instead - the maximum achievable "
               f"capital here is ~{meta['achieved_max_m']:,.1f}M EUR."
               if not meta["target_reached"]
               else "The capital display follows the sidebar toggle."))

        pins = st.session_state.get("pins", [])
        overlay = [(pin_label(p), p.get("pts", 0.0), p.get("cap", 0.0))
                   for p in pins]
        fig = build_scatter(scenarios_df, overlay=overlay)
        st.plotly_chart(fig, key="pareto_scatter", on_select="rerun",
                        selection_mode="points", width="stretch")

        sel = st.session_state.get("pareto_scatter")
        clicked_id = _clicked_scenario_id(sel)
        if clicked_id is not None:
            st.session_state["sel_sid"] = int(clicked_id)

        # Scenario ids restart at 0 for every club/season/target combo, so a "sel_sid"
        # remembered from a previous analysis would silently point at the wrong
        # scenario. Reset it whenever the analysis context changes.
        analysis_ctx = (int(team_id), int(season_id), int(target_raise_m),
                        int(max_players), tuple(sorted(min_depth.items())))
        if st.session_state.get("_analysis_ctx") != analysis_ctx:
            st.session_state["_analysis_ctx"] = analysis_ctx
            st.session_state["sel_sid"] = None

        if not scenarios_df[scenarios_df["pareto_optimal"]].empty:
            default_sid = int(scenarios_df[scenarios_df["pareto_optimal"]]
                              .sort_values("predicted_points_delta")["scenario_id"].iloc[0])
        elif len(scenarios_df):
            default_sid = int(scenarios_df.sort_values(
                "predicted_points_delta")["scenario_id"].iloc[0])
        else:
            default_sid = 0

        options = scenarios_df.sort_values(["pareto_optimal", capital_col],
                                           ascending=[False, False])
        labels = [format_scenario_label(r) for _, r in options.iterrows()]
        ids = options["scenario_id"].to_numpy()
        current = st.session_state.get("sel_sid") or default_sid
        match = np.where(ids == current)[0]
        cur_idx = int(match[0]) if len(match) else 0
        picked = st.selectbox("Scenario to inspect (or click a point on the chart)",
                              options=range(len(labels)), index=cur_idx,
                              format_func=lambda i: labels[i])
        sid = int(ids[picked])
        st.session_state["sel_sid"] = sid

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        row = scenarios_df[scenarios_df["scenario_id"] == sid].iloc[0]
        r1.metric("Players sold", int(row["n_players_sold"]))
        r2.metric("Capital (flat)", f"{row['revenue_eur_m']:,.1f}M EUR")
        r3.metric("Model net impact",
                  f"{row.get('net_eur_m', row['revenue_eur_m']):,.1f}M EUR")
        r4.metric("Predicted points lost", f"{-float(row['predicted_points_delta']):.1f}")
        r5.metric("Relegation risk change", f"{row['risk_change_pp']:+.2f} pp")
        r6.metric("Depth constraints OK", "Yes" if bool(row["depth_ok"]) else "No")

        if st.button("Pin this scenario to Compare", key="pin_planner",
                     use_container_width=True):
            pins_ = st.session_state["pins"]
            pins_.append({
                "_uid": next_uid, "kind": "pre", "sid": int(sid),
                "n": int(row["n_players_sold"]), "cap": float(row[capital_col]),
                "pts": float(row["predicted_points_delta"]),
                "rel": float(row["predicted_relegation_risk_change"]),
                "label": f"#{sid}"})
            st.session_state["pin_uid"] = next_uid + 1
            st.session_state["pins"] = pins_
            st.toast("Pinned scenario #%d" % sid)

        st.subheader("Players sold")
        sold = players_df[players_df["scenario_id"] == sid].copy()
        fin = load_financials()
        try:
            fin_season = fin[fin["season_id"] == season_id]
            sold = sold.merge(fin_season[["player_id", "purchase_fee_m",
                                          "last_sale_fee_m", "is_la_fabrica_academy"]],
                              on="player_id", how="left")
        except Exception:
            pass
        sold_disp = sold.copy()
        if "market_value" in sold_disp.columns:
            sold_disp["market_value"] = sold_disp["market_value"] / 1e6
        st.dataframe(sold_disp.rename(columns={
            "short_name": "Player", "position_group": "Pos", "age": "Age",
            "market_value": "Market value (EUR M)", "expected_gc": "Expected GC",
            "decline_risk_20pct": "Decline risk (20%)",
            "purchase_fee_m": "Purchase fee (EUR M)",
            "last_sale_fee_m": "Last sale fee (EUR M)",
            "is_la_fabrica_academy": "La Fabrica"}),
            hide_index=True, width="stretch",
            column_config={
                "Market value (EUR M)": st.column_config.NumberColumn(format="%.1f"),
                "Expected GC": st.column_config.NumberColumn(format="%.2f"),
                "Decline risk (20%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Purchase fee (EUR M)": st.column_config.NumberColumn(format="%.1f"),
                "Last sale fee (EUR M)": st.column_config.NumberColumn(format="%.1f"),
                "La Fabrica": st.column_config.CheckboxColumn()})

        with st.expander("Before / after squad capacity"):
            st.dataframe(before_after_rows(base_state, row),
                         hide_index=True, width="stretch",
                         column_config={
                             "Before": st.column_config.NumberColumn(format="%.2f"),
                             "After": st.column_config.NumberColumn(format="%.2f"),
                             "Delta": st.column_config.NumberColumn(format="%.2f")})

        with st.expander("Full scenario table"):
            view = scenarios_df[[
                "scenario_id", "n_players_sold", "revenue_eur_m", "net_eur_m",
                "predicted_points_baseline", "predicted_points_delta",
                "predicted_relegation_risk_before", "predicted_relegation_risk_after",
                "risk_change_pp", "expected_gc_retained_pct", "depth_ok",
                "pareto_optimal"]].rename(columns={
                    "scenario_id": "Scenario", "n_players_sold": "Players",
                    "revenue_eur_m": "Capital flat (EUR M)",
                    "net_eur_m": "Net impact (EUR M)",
                    "predicted_points_baseline": "xP baseline",
                    "predicted_points_delta": "xP delta",
                    "predicted_relegation_risk_before": "Rel risk before",
                    "predicted_relegation_risk_after": "Rel risk after",
                    "risk_change_pp": "Risk change (pp)",
                    "expected_gc_retained_pct": "GC retained",
                    "depth_ok": "Depth OK", "pareto_optimal": "Pareto"})
            st.dataframe(view, hide_index=True, width="stretch",
                         column_config={
                             "Capital flat (EUR M)": st.column_config.NumberColumn(format="%.1f"),
                             "Net impact (EUR M)": st.column_config.NumberColumn(format="%.1f"),
                             "xP baseline": st.column_config.NumberColumn(format="%.1f"),
                             "xP delta": st.column_config.NumberColumn(format="%.1f"),
                             "Rel risk before": st.column_config.NumberColumn(format="%.4f"),
                             "Rel risk after": st.column_config.NumberColumn(format="%.4f"),
                             "Risk change (pp)": st.column_config.NumberColumn(format="%.2f"),
                             "GC retained": st.column_config.NumberColumn(format="%.1f%%"),
                             "Pareto": st.column_config.CheckboxColumn()})

    # ---------------- Scenario Builder ----------------
    with tab_builder:
        st.subheader("Custom scenario builder (sell + buy)")
        st.caption("Pick any players to sell and any affordable signing to add. "
                   "Scored live with the same engine as the 3000-scenario batch "
                   "path - no pre-baked combinations required.")
        squad = squad_roster(sq, team_id, season_id)
        if squad.empty:
            st.info("No squad data for this club/season in the builder dataset.")
            st.stop()
        base_state = scf.squad_state(squad)
        pool = candidate_roster(sq, team_id, season_id, pricing)

        sell_opts = squad.sort_values("market_value_eur", ascending=False)
        sell_lab = (sell_opts["short_name"].fillna("?") + "  [" +
                    sell_opts["position_group"] + " | " +
                    (sell_opts["market_value_eur"] / 1e6).round(0).astype(str) + "M]")
        sold_sel = st.multiselect(
            "Players to sell",
            options=range(len(sell_opts)), format_func=lambda i: sell_lab.iloc[i],
            help="Max players per the sidebar limit.")
        sold_ids = [int(sell_opts.iloc[i]["player_id"]) for i in sold_sel]
        st.caption(f"Selected to sell: {len(sold_ids)} players | "
                   f"flat value {sum(float(sell_opts.iloc[i]['market_value_eur']) for i in sold_sel) * SALE_MULTIPLIER / 1e6:,.1f}M EUR")

        budget_default = float(sum(sell_opts.iloc[i]["market_value_eur"] for i in sold_sel)
                               * SALE_MULTIPLIER)
        budget = st.slider("Available signing budget (EUR M)", 2, 400,
                           int(min(max(budget_default / 2e6, 2), 400)), 1)
        affordable = pool[pool["purchase_fee_est"] <= budget * 1e6]
        buy_opts = affordable.sort_values(["stat_total_goals"
                                           if "stat_total_goals" in affordable
                                           else "expected_gc",
                                           "market_value_eur"],
                                          ascending=False)
        buy_lab = (buy_opts["short_name"].fillna("?") + "  [" +
                   buy_opts["position_group"] + " | " +
                   buy_opts["team_name"].astype(str).str.slice(0, 12) + "] " +
                   (buy_opts["market_value_eur"] / 1e6).round(1).astype(str) + "M")
        bought_sel = st.multiselect(
            f"Signings to add (from other {season_option} LaLiga clubs)",
            options=range(len(buy_opts)), format_func=lambda i: buy_lab.iloc[i])
        bought_ids = [int(buy_opts.iloc[i]["player_id"]) for i in bought_sel]

        st.caption(f"{len(buy_opts)} affordable candidates with budget "
                   f"{budget:.0f}M (purchase = current market value).")

        if not sold_ids:
            st.info("Select at least one player to sell to score a scenario.")
            st.stop()

        res = score_sell_buy(sq, team_id, season_id, sold_ids, bought_ids, pool,
                             base_state, model, pricing)

        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Capital (flat 0.95xMV)", f"{res['flat_raise'] / 1e6:,.1f}M EUR")
        b2.metric("Model sale fee", f"{res['model_fee'] / 1e6:,.1f}M EUR")
        b3.metric("Wages+contract avoided (est)",
                  f"{res['wage_savings'] / 1e6:,.1f}M EUR")
        b4.metric("Net cash impact", f"{res['net_impact'] / 1e6:,.1f}M EUR")
        b5.metric("Points lost", f"{res['points_delta']:+.1f}")
        b6.metric("Relegation swing", f"{res['relegation_change']:+.2%}")
        if not is_rm:
            st.caption("Learned transfer-fee/wage model is Real Madrid-only - the "
                       "'Model sale fee' here is the flat 0.95x market value proxy.")

        if res["buy_names"]:
            st.markdown(f"**Signing:** {' + '.join(f'{n} ({t})' for n, t in zip(res['buy_names'], res['buy_teams']))} "
                        f"for **{res['purchase'] / 1e6:,.1f}M EUR** | "
                        f"net capital after buy **{res['net_capital'] / 1e6:,.1f}M**")

        if not res["depth_ok"]:
            st.warning("Depth check: this sell set breaks minimum squad depth "
                       f"(post-sale counts {res['post_counts']}; minima "
                       f"{DEFAULT_MIN_DEPTH}). Buying helps only if it restores the "
                       "count. The scenario is shown, not blocked - but do not "
                       "submit this to the board as-is.")
        else:
            st.success(f"Depth OK after swap: {res['post_counts']}")

        # Glue: does the sell-set match a committed RM 24/25 scenario with a
        # recommended buy? The committed batch artifacts are RM-only, so the glue
        # is gated to that exact club-season.
        help_sid = None
        if is_rm and season_id == SEASON_ID_2024_25:
            try:
                signings = load_scenario_signings()
                sply = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "scenario_players.csv")
                pbs = sply.groupby("scenario_id")["player_id"].apply(set).to_dict()
                match = match_batch_signing(sold_ids, signings, pbs)
                if match:
                    help_sid = match["scenario_id"]
                    st.info(
                        f"Batch engine (scenario #{match['scenario_id']}) would recommend "
                        f"**{match['name']}** ({match['team']}, {match['position']}, "
                        f"{match['purchase_m']:.0f}M) with up to "
                        f"{match['recovery']:.1f} pts recovery - compare against your picks.")
            except Exception:
                pass

        if st.button("Pin this custom scenario to Compare",
                     use_container_width=True, key="pin_builder"):
            pins_ = st.session_state["pins"]
            pins_.append({
                "_uid": next_uid, "kind": "custom",
                "sell_names": res["sold_names"], "buy_names": res["buy_names"],
                "buy_teams": res["buy_teams"],
                "cap": res["model_fee"] / 1e6, "net": res["net_impact"] / 1e6,
                "purchase": res["purchase"] / 1e6, "pts": res["points_delta"],
                "rel": res["relegation_change"], "depth_ok": res["depth_ok"]})
            st.session_state["pin_uid"] = next_uid + 1
            st.session_state["pins"] = pins_
            st.session_state["sel_sid"] = int(help_sid or 0)
            st.toast("Pinned custom scenario")

        with st.expander("Per-player economics of this sale set"):
            if is_rm:
                pm = pricing[(pricing["player_id"].isin(sold_ids)) &
                             (pricing["season_id"] == season_id)][[
                                 "short_name", "position_group", "market_value_eur",
                                 "predicted_sale_fee", "annual_wage_eur",
                                 "contract_years_remaining"]]
                if pm.empty:
                    pm = squad[squad["player_id"].isin(sold_ids)][
                        ["short_name", "position_group", "market_value_eur"]].copy()
                    pm["predicted_sale_fee"] = pm["market_value_eur"] * SALE_MULTIPLIER
                    pm["annual_wage_eur"] = np.nan
                    pm["contract_years_remaining"] = DEFAULT_MIN_DEPTH.get("MF", 0)
            else:
                pm = squad[squad["player_id"].isin(sold_ids)][
                    ["short_name", "position_group", "market_value_eur"]].copy()
                pm["predicted_sale_fee"] = pm["market_value_eur"] * SALE_MULTIPLIER
                pm["annual_wage_eur"] = np.nan
                pm["contract_years_remaining"] = DEFAULT_MIN_DEPTH.get("MF", 0)
            view = pm.copy()
            view["wage_contract_total"] = view["annual_wage_eur"] * view["contract_years_remaining"]
            st.dataframe(view.rename(columns={
                "short_name": "Player", "position_group": "Pos",
                "market_value_eur": "Market value (EUR)",
                "predicted_sale_fee": "Model sale fee (EUR)",
                "annual_wage_eur": "Annual wage est (EUR)",
                "contract_years_remaining": "Contract y. (est)",
                "wage_contract_total": "Wages+contract avoided (EUR)"}),
                hide_index=True, width="stretch",
                column_config={
                    "Market value (EUR)": st.column_config.NumberColumn(format="%.0f"),
                    "Model sale fee (EUR)": st.column_config.NumberColumn(format="%.0f"),
                    "Annual wage est (EUR)": st.column_config.NumberColumn(format="%.0f"),
                    "Wages+contract avoided (EUR)": st.column_config.NumberColumn(format="%.0f")})

    # ---------------- Compare ----------------
    with tab_compare:
        render_pinned_compare()
        if st.session_state.get("pins"):
            if st.button("Clear all pinned", use_container_width=True):
                st.session_state["pins"] = []
                st.rerun()

    # ---------------- La Fabrica ----------------
    with tab_lafab:
        if not is_rm:
            st.info("Detailed La Fabrica academy metrics and club financial "
                    "balance sheets are currently available for Real Madrid only.")
        else:
            render_la_fabrica(load_la_fabrica_aggregate(), load_la_fabrica_cases(),
                              load_financials())

    # ---------------- Trust ----------------
    with tab_trust:
        render_trust_tab()


def _in_streamlit() -> bool:
    try:
        from streamlit import runtime
        return bool(runtime.exists())
    except Exception:
        return False


def _clicked_scenario_id(sel):
    """Extract the clicked point's scenario id from a plotly selection event,
    tolerant of both the dict-based and object-based Streamlit event APIs."""
    if sel is None:
        return None
    points = None
    try:
        points = sel.selection.points
    except Exception:
        try:
            points = (sel.get("selection", {}).get("points")
                      if hasattr(sel, "get") else None)
        except Exception:
            points = None
    if not points:
        return None
    p0 = points[0]
    cd = p0.get("customdata") if hasattr(p0, "get") else getattr(p0, "customdata", None)
    if cd is None:
        return None
    try:
        return int(cd[0])
    except (TypeError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Headless self-check / entry point
# ---------------------------------------------------------------------------
def run_selfcheck():
    import time
    t0 = time.perf_counter()
    print("selfcheck: precomputed-chain parity + custom cases + multi-club coverage")
    model = load_club_model()
    sq = load_squad_frame()
    pricing = load_pricing_master()
    squad = squad_roster(sq, 675, 189974)
    if squad.empty:
        print("SELFCHECK FAIL: no squad")
        return 1
    base_state = scf.squad_state(squad)

    # 0) committed-chain parity (batch engine + shared scorer + pareto)
    result = compute_analysis(675, 189974, 20_000_000, 3,
                              {"GK": 2, "DF": 6, "MF": 6, "FW": 4})
    if result is None:
        print("SELFCHECK FAIL: compute_analysis returned None")
        return 1
    scenarios_df, players_df, _, _ = result
    if len(scenarios_df) != 3000:
        print(f"SELFCHECK FAIL: expected 3000 scenarios, got {len(scenarios_df)}")
        return 1
    frontier = scenarios_df[scenarios_df["pareto_optimal"]].sort_values(
        "predicted_points_delta")
    print(f"  [parity]  scenarios={len(scenarios_df)} pareto={int(scenarios_df['pareto_optimal'].sum())} "
          f"players_rows={len(players_df)}")
    for _, r in frontier.head(2).iterrows():
        print(f"    frontier #{int(r['scenario_id'])} sold={int(r['n_players_sold'])} "
              f"raise={r['revenue_eur_m']:.1f}M pts={r['predicted_points_delta']:+.1f}")

    # helper for the custom cases
    def score(ids, buys):
        return score_sell_buy(sq, 675, 189974, ids, buys, candidate_roster(
            sq, 675, 189974, pricing), base_state, model, pricing)

    # 1) custom sell-only: single star sale (diacritic-insensitive name match)
    import unicodedata
    def _no_acc(s: str) -> str:
        return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    vin = squad[squad["short_name"].apply(lambda x: "vinicius" in _no_acc(x).lower())]
    if vin.empty:
        print("SELFCHECK FAIL: Vinicius not found")
        return 1
    r1 = score([int(vin.iloc[0]["player_id"])], [])
    assert r1["depth_ok"] is True
    assert r1["model_fee"] > 0 and r1["flat_raise"] > 0
    assert r1["points_delta"] < 0
    print(f"  [custom sell-only] sell={r1['sold_names'][0]} flat={r1['flat_raise']/1e6:.0f}M "
          f"model_fee={r1['model_fee']/1e6:.0f}M net={r1['net_impact']/1e6:.0f}M "
          f"pts={r1['points_delta']:+.1f} depth_ok={r1['depth_ok']}")

    # 2) custom sell+buy: same star sale + a signing
    pool = candidate_roster(sq, 675, 189974, pricing)
    buy_cand = pool[pool["position_group"] == "FW"].sort_values(
        "market_value_eur", ascending=False)
    bc = buy_cand.iloc[0]
    r2 = score([int(vin.iloc[0]["player_id"])], [int(bc["player_id"])])
    assert r2["depth_ok"] is True
    assert r2["purchase"] > 0 and r2["net_capital"] < r2["model_fee"]
    assert r2["points_delta"] > r1["points_delta"]  # buying an attacker recovers points
    print(f"  [custom sell+buy]  buy={r2['buy_names'][0]} ({r2['buy_teams'][0]}) "
          f"purchase={r2['purchase']/1e6:.0f}M net_capital={r2['net_capital']/1e6:.0f}M "
          f"pts={r2['points_delta']:+.1f} (sell-only {r1['points_delta']:+.1f})")

    # 3) depth-violation: sell 2 GKs -> remaining < 2 -> warning path, no crash
    gks = squad[squad["position_group"] == "GK"].head(2)
    r3 = score([int(p) for p in gks["player_id"]], [])
    assert r3["depth_ok"] is False
    print(f"  [custom depth-violation] sell 2 GKs -> depth_ok={r3['depth_ok']} "
          f"post_counts={r3['post_counts']} (warning shown, not blocked)")

    # 4) multi-club regression: every supported club/season must compute without
    #    crashing, show NO Real-Madrid batch contamination, keep the flat
    #    market-value proxy for non-RM learned views, and guardrail impossible
    #    targets instead of raising KeyError / showing a blank screen.
    def _team_id(name, sid):
        cat = load_club_catalog()
        rows = cat[cat["season_id"] == sid]
        hit = rows[rows["team_name"].apply(
            lambda x: name.lower() in _no_acc(str(x)).lower())]
        return int(hit["team_id"].iloc[0])

    n_club_scenarios = 0
    club_cases = [
        ("Barcelona", SEASON_ID_2024_25),  # FW=3 baseline -> depth guardrail path
        ("Barcelona", 189012),             # FW=4 == default minimum floor
        ("Atletico", 189974),              # MF=3 baseline -> depth guardrail path
        ("Cadiz", 189012),                 # smaller club (not in 2024/25)
        ("Rayo", 189974),                  # mid-size club across a further season
    ]
    for cname, sid in club_cases:
        tm = _team_id(cname, sid)
        res = compute_analysis(tm, sid, 20_000_000, 3,
                               {"GK": 2, "DF": 6, "MF": 6, "FW": 4})
        if res is None:
            print(f"  FAIL club {cname} {sid}: compute_analysis returned None")
            return 1
        sdf, pdf, bs, meta = res
        if int(sdf["predicted_sale_fee"].notna().sum()):
            print(f"  FAIL club {cname} {sid}: RM batch fee data leaked onto non-RM club")
            return 1
        if not sdf.empty:
            leak = float((sdf["net_financial_impact"].fillna(sdf["raise_value"])
                          - sdf["raise_value"]).abs().max())
            if leak > 1.0:
                print(f"  FAIL club {cname} {sid}: net impact != flat proxy (leak={leak:.0f})")
                return 1
            n_club_scenarios += len(sdf)
        # extreme target: never crash, and if any scenario set exists the target
        # must be flagged as NOT reached (blank-screen guardrail)
        extreme = max(float(bs.get("total_market_value") or 0) * 1.5, 200_000_000)
        res2 = compute_analysis(tm, sid, extreme, 3,
                                {"GK": 2, "DF": 6, "MF": 6, "FW": 4})
        if res2 is None:
            print(f"  FAIL club {cname} {sid}: extreme-target run crashed")
            return 1
        sdf2, _p2, _b2, meta2 = res2
        if not sdf2.empty:
            if meta2["target_reached"]:
                print(f"  FAIL club {cname} {sid}: impossible target mis-flagged as reached")
                return 1
            n_club_scenarios += len(sdf2)
        # builder path for a non-RM club: roster + buy pool populate, learned fee
        # stays disabled (flat 0.95xMV proxy), no exceptions
        ro = squad_roster(sq, tm, sid)
        pool = candidate_roster(sq, tm, sid, pricing)
        if ro.empty or pool.empty:
            print(f"  FAIL club {cname} {sid}: builder roster/pool empty")
            return 1
        star = ro.sort_values("market_value_eur", ascending=False).iloc[0]
        r = score_sell_buy(sq, tm, sid, [int(star["player_id"])], [], pool,
                           scf.squad_state(ro), model, pricing)
        if abs(r["model_fee"] - r["flat_raise"]) > 1.0 or r["wage_savings"] != 0.0:
            print(f"  FAIL club {cname} {sid}: non-RM learned fee enabled unexpectedly")
            return 1
        print(f"    {cname} {sid}: {len(sdf)} scenarios (normal) / "
              f"{len(sdf2)} closest-set (extreme {extreme/1e6:.0f}M), flat-only, "
              f"guardrail target_reached={meta2['target_reached']}")
    print(f"  [clubs] covers {len(club_cases)} club-seasons across 4 non-RM clubs "
          f"(Barcelona, Atletico Madrid, Cadiz, Rayo Vallecano); "
          f"{n_club_scenarios} scenarios generated total")

    print(f"  elapsed={time.perf_counter() - t0:.1f}s")
    print("SELFCHECK OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        sys.exit(run_selfcheck())
    elif _in_streamlit():
        main()
    else:
        print("Run this dashboard with:  streamlit run src/dashboard/app.py")
        print("Or verify the compute chain with:  python src/dashboard/app.py --selfcheck")