# -*- coding: utf-8 -*-
"""
scenario_scoring.py - Single source of truth for ClubSportingModel scenario scoring.

Every consumer (batch scenario_pareto path, buy-side engine, Streamlit custom
scenario builder, headless selfcheck) scores a scenario through the SAME
vectorized replica of ClubSportingModel.predict_scenario. The vectorized path
was verified 1:1 against the per-row path on the 3000 committed scenarios
(max abs diff = 0.0 on all six prediction columns).

Empty DataFrames (no feasible scenarios under strict depth constraints) return
an empty scored frame with the expected columns instead of raising a KeyError.

Module is intentionally dependency-light (numpy / pandas / scipy only) so it
can be reused outside the dashboard process.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from optimization import scenario_pareto as sp
from models.club_sporting_model import ATTACK_FEATURES, DEFENCE_FEATURES

GOAL_GRID = np.arange(0, 8)
_WIN = GOAL_GRID[:, None] > GOAL_GRID[None, :]
_DRAW = GOAL_GRID[:, None] == GOAL_GRID[None, :]

SCORED_COLS = [
    "predicted_points_baseline",
    "predicted_points_after",
    "predicted_points_delta",
    "predicted_relegation_risk_before",
    "predicted_relegation_risk_after",
    "predicted_relegation_risk_change",
]


def expected_points_vec(model, gf, ga, matches: int = 38):
    """Vectorized expected-points Poisson simulator (matches per-row predict)."""
    gf = np.asarray(gf, dtype=float)
    ga = np.asarray(ga, dtype=float)
    lag = model.league_avg_goals
    lam_for = lag / matches * (gf / lag)
    lam_ag = lag / matches * (ga / lag)
    p_for = poisson.pmf(GOAL_GRID[None, :], lam_for[:, None])
    p_ag = poisson.pmf(GOAL_GRID[None, :], lam_ag[:, None])
    p_win = np.einsum("ni,nj,ij->n", p_for, p_ag, _WIN)
    p_draw = np.einsum("ni,nj,ij->n", p_for, p_ag, _DRAW)
    return (3.0 * p_win + 1.0 * p_draw) * matches


def feature_matrix(df: pd.DataFrame, caps, prefix: str) -> np.ndarray:
    """Resolve prefixed (base_/post_) capacity columns into a float matrix,
    falling back to bare feature names and filling any missing feature with
    0.0 so scoring never raises a KeyError on partial schemas."""
    out = np.zeros((len(df), len(caps)), dtype=float)
    for j, c in enumerate(caps):
        col = f"{prefix}{c}"
        if col not in df.columns:
            col = c
        if col in df.columns:
            out[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
    return out


def score_scenarios_fast(model, df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized drop-in for scenario_pareto.apply_model_to_scenarios.

    The six output columns exactly match ClubSportingModel.predict_scenario:
    predicted_points_baseline / after / delta + relegation-risk before/after/change.
    """
    if df is None or df.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in SCORED_COLS})

    caps = sp.CAPACITY_FEATURES
    base = feature_matrix(df, caps, "base_")
    post = feature_matrix(df, caps, "post_")
    ai = [caps.index(c) for c in ATTACK_FEATURES]
    di = [caps.index(c) for c in DEFENCE_FEATURES]
    gf_base = model.attack_model.predict(model.attack_scaler.transform(base[:, ai]))
    gf_after = model.attack_model.predict(model.attack_scaler.transform(post[:, ai]))
    ga_base = model.defence_model.predict(model.defence_scaler.transform(base[:, di]))
    lag = model.league_avg_goals
    xp_before = expected_points_vec(model, gf_base, ga_base)
    att_after = expected_points_vec(model, gf_after, np.full_like(gf_after, lag))
    att_before = expected_points_vec(model, gf_base, np.full_like(gf_base, lag))
    delta = att_after - att_before
    xp_after = xp_before + delta
    risk_b = model.relegation_model.predict_proba(xp_before[:, None])[:, 1]
    risk_a = model.relegation_model.predict_proba(xp_after[:, None])[:, 1]
    return pd.DataFrame({
        "predicted_points_baseline": np.round(xp_before, 1),
        "predicted_points_after": np.round(xp_after, 1),
        "predicted_points_delta": np.round(delta, 1),
        "predicted_relegation_risk_before": np.round(risk_b, 4),
        "predicted_relegation_risk_after": np.round(risk_a, 4),
        "predicted_relegation_risk_change": np.round(risk_a - risk_b, 4),
    })


COUNTERFACTUAL_COLS = [
    "squad_size_post", "total_market_value_post", "avg_market_value_post",
    "avg_age_post", "total_gc_post", "total_expected_gc_post",
    "avg_decline_risk_post", "goal_concentration_hhi_post",
    "top_scorer_share_post", "count_GK_post", "count_DF_post",
    "count_MF_post", "count_FW_post", "expected_gc_retained_pct", "depth_ok",
]


def scenario_schema_columns() -> list:
    """Canonical non-scored column set for a scenarios DataFrame, so an empty
    result (zero feasible scenarios) still carries the standard schema."""
    caps = sp.CAPACITY_FEATURES
    return (["scenario_id", "team_id", "season_id", "n_players_sold",
             "raise_value", "total_sold_value", "raise_excess"]
            + [f"base_{c}" for c in caps]
            + [f"post_{c}" for c in caps]
            + COUNTERFACTUAL_COLS)


def build_scenario_row(base_state: dict, post_state: dict,
                       meta: dict) -> dict:
    """Build a single scored-scenario row (base_*/post_* features + schema cols)
    from squad states and an arbitrary meta dict (scenario_id, raise_value, ...)."""
    post = dict(post_state)
    base = dict(base_state)
    row = dict(meta)
    caps = sp.CAPACITY_FEATURES
    from simulation import squad_counterfactual as scf
    bf = scf.to_model_features(base)
    pf = scf.to_model_features(post)
    for k in caps:
        row[f"base_{k}"] = bf.get(k, 0) or 0
    for k in caps:
        row[f"post_{k}"] = pf.get(k, 0) or 0
    return row