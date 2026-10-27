"""
club_sporting_model.py
=======================
Structural Club-Level Sporting Outcome Predictor.

WHY STRUCTURAL (not raw ML):
  club_season_dataset.csv has only N=34 valid t -> t+1 transitions. Naive
  XGBoost/DNN would violently overfit. Instead we use a Structural Expected
  Points (xP) framework:

    1. Squad Capacity -> Goals mapping (calibrated on ALL 60 team-seasons,
       not just the 34 transitions):
       - GF_cap ~ f(retained goal contributions, shots, market value)
       - GA_cap ~ f(retained defensive/market-value features)
    2. Expected Points via a Poisson match simulator:
       - attack/defence strength vs the season's league-average opponent
       - P(W)/P(D)/P(L) from a Poisson outcome model -> xP per 38 matches
    3. Relegation risk from predicted xP (logistic calibration over the
       60 team-seasons observing actual relegation status).

  For a SALE SCENARIO the model is applied to the POST-DEPARTURE squad
  features and compared to the BASELINE squad features, yielding
  predicted_points_delta and relegation-risk delta. Because the capacity
  mapping is monotone increasing in goal contribution, losing goals MUST
  lower predicted points (validated by the monotonicity check).

CALIBRATION NOTES:
  - total_goal_contributions predicts realised team GF at R2 ~ 0.976, so the
    attack mapping is essentially deterministic given squad GC. We therefore
    keep the attack slope fixed-like and calibrate only its scale/shift on
    the pooled 60 team-seasons (robust).
  - GA is harder (R2 ~ 0.26); we use a Ridge-regularised mapping on
    defensively-relevant features and pool all seasons.

Outputs:
  - models/club_sporting_model.pkl
  - data/processed/club_model_evaluation.json
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(r"C:\football")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

SEASON_IDS = [188124, 189012, 189974]

# Canonical feature groups for squad capacity -> goals mapping.
# These must be computable for BOTH the full squad and an arbitrary
# post-departure squad subset (they are simple per-player aggregates).
ATTACK_FEATURES = [
    "total_goal_contributions",   # goals + assists (near-deterministic -> GF)
    "total_shots",
    "total_assists",
]
DEFENCE_FEATURES = [
    "total_market_value",
    "total_interceptions",
    "count_DF",
    "squad_minutes_utilization",
    "count_GK",
    "avg_age",
]

# League-average goals per match, per season (from observed GF/GA)
LEAGUE_AVG_GOALS = {
    188124: 47.75,
    189012: 50.25,
    189974: 49.75,
}


class ClubSportingModel:
    """
    Structural expected-points predictor.

    Components trained on the pooled 60 team-seasons:
      attack_model : squad attack features -> GF_cap (Ridge)
      defence_model: squad defence features -> GA_cap (Ridge)
      relegation_model: xP -> relegation probability (Logistic)

    Prediction pipeline for a squad feature vector:
      attack_str = GF_cap / league_avg_goals
      def_weak   = GA_cap / league_avg_goals
      lambda_for, lambda_against vs average opponent
      Poisson -> xP
    """

    def __init__(self, season_id: int):
        self.season_id = season_id
        self.attack_model = None
        self.defence_model = None
        self.relegation_model = None
        self.attack_scaler = StandardScaler()
        self.defence_scaler = StandardScaler()
        self.league_avg_goals = LEAGUE_AVG_GOALS.get(season_id, 49.9)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, cs: pd.DataFrame):
        """Fit capacity + relegation models on pooled team-seasons."""
        # ---- Attack mapping: squad features -> GF ----
        Xa = cs[ATTACK_FEATURES].fillna(0)
        ya = cs["team_gf"].values.astype(float)
        self.attack_scaler.fit(Xa)
        Xa_s = self.attack_scaler.transform(Xa)
        self.attack_model = Ridge(alpha=5.0)
        self.attack_model.fit(Xa_s, ya)
        attack_r2 = self.attack_model.score(Xa_s, ya)
        log.info("Attack model (squad -> GF): R2=%.3f", attack_r2)

        # ---- Defence mapping: squad features -> GA ----
        Xd = cs[DEFENCE_FEATURES].fillna(0)
        yd = cs["team_ga"].values.astype(float)
        self.defence_scaler.fit(Xd)
        Xd_s = self.defence_scaler.transform(Xd)
        self.defence_model = Ridge(alpha=8.0)
        self.defence_model.fit(Xd_s, yd)
        defence_r2 = self.defence_model.score(Xd_s, yd)
        log.info("Defence model (squad -> GA): R2=%.3f", defence_r2)

        # ---- Relegation model: xP -> relegation probability ----
        # Relegation = rank >= 18 (bottom 3)
        cs = cs.copy()
        cs["relegated_flag"] = cs["team_league_rank"].ge(18).astype(int)
        # xP baseline from actual points (38-game, points/38*38)
        xp = cs["team_points"].values.astype(float).reshape(-1, 1)
        self.relegation_model = LogisticRegression(C=0.5)
        self.relegation_model.fit(xp, cs["relegated_flag"].values)
        rel_acc = self.relegation_model.score(xp, cs["relegated_flag"].values)
        log.info("Relegation model (xP -> relegation): acc=%.3f", rel_acc)

        self._attack_r2 = attack_r2
        self._defence_r2 = defence_r2

    # ------------------------------------------------------------------
    # Capacity predictions
    # ------------------------------------------------------------------
    def predict_gf(self, features: dict) -> float:
        Xa = pd.DataFrame([features])[ATTACK_FEATURES].fillna(0)
        Xa_s = self.attack_scaler.transform(Xa)
        return float(self.attack_model.predict(Xa_s)[0])

    def predict_ga(self, features: dict) -> float:
        Xd = pd.DataFrame([features])[DEFENCE_FEATURES].fillna(0)
        Xd_s = self.defence_scaler.transform(Xd)
        return float(self.defence_model.predict(Xd_s)[0])

    # ------------------------------------------------------------------
    # Expected points via Poisson simulator
    # ------------------------------------------------------------------
    def expected_points(self, gf_cap: float, ga_cap: float,
                        matches: int = 38) -> float:
        """Simulate an entire season vs an average LaLiga opponent."""
        attack_str = gf_cap / self.league_avg_goals
        def_weak = ga_cap / self.league_avg_goals
        # Expected goals per match vs average opponent
        lambda_for = self.league_avg_goals / matches * attack_str
        lambda_against = self.league_avg_goals / matches * def_weak

        # Poisson outcome probabilities (independent Poisson goals)
        go_range = np.arange(0, 8)
        p_for = poisson.pmf(go_range, lambda_for)
        p_ag = poisson.pmf(go_range, lambda_against)
        p_win = 0.0
        p_draw = 0.0
        for i, pf in enumerate(p_for):
            for j, pa in enumerate(p_ag):
                if i > j:
                    p_win += pf * pa
                elif i == j:
                    p_draw += pf * pa
        p_win = min(p_win, 1.0)
        p_draw = min(p_draw, 1.0)
        return float((3 * p_win + 1 * p_draw) * matches)

    def predict_points(self, features: dict) -> float:
        gf = self.predict_gf(features)
        ga = self.predict_ga(features)
        return self.expected_points(gf, ga)

    def predict_relegation_risk(self, xp: float) -> float:
        p = self.relegation_model.predict_proba(np.array([[xp]]))[0]
        return float(p[1])

    # ------------------------------------------------------------------
    # Scenario-level predictions
    # ------------------------------------------------------------------
    def predict_scenario(self, baseline_features: dict,
                         after_features: dict) -> dict:
        """
        Scenario predictions for a departure set.

        `predicted_points_baseline` is the full-model xP of the intact squad.
        `predicted_points_delta` is the ROBUST attack-driven delta (computed
        from lost goal-production capacity, holding defence at the league
        average). Using the attack side avoids the small-sample defence
        generalisation noise that dominates when only 1-3 players depart.
        `predicted_points_after` is baseline + attack delta (self-consistent).
        """
        xp_before = self.predict_points(baseline_features)
        delta = self.attack_only_delta_points(baseline_features, after_features)
        xp_after = xp_before + delta
        risk_before = self.predict_relegation_risk(xp_before)
        risk_after = self.predict_relegation_risk(xp_after)
        return {
            "predicted_points_baseline": round(xp_before, 1),
            "predicted_points_after": round(xp_after, 1),
            "predicted_points_delta": round(delta, 1),
            "predicted_relegation_risk_before": round(risk_before, 4),
            "predicted_relegation_risk_after": round(risk_after, 4),
            "predicted_relegation_risk_change": round(risk_after - risk_before, 4),
        }

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def attack_only_delta_points(self, baseline_features: dict,
                                 after_features: dict,
                                 matches: int = 38) -> float:
        """
        Attack-driven points delta: recompute xP for the predicted GOAL
        OUTPUT (GF_cap) of the full vs post squad against a league-average
        opponent, holding defence at league average. Robust and monotone.
        """
        gf_partial = lambda gf_cap: self.expected_points(
            gf_cap, self.league_avg_goals, matches)
        gf_base = self.predict_gf(baseline_features)
        gf_after = self.predict_gf(after_features)
        return gf_partial(gf_after) - gf_partial(gf_base)


# ----------------------------------------------------------------------
# Feature extraction from a squad aggregate row (dict of aggregate values)
# ----------------------------------------------------------------------
def extract_squad_features(row: dict) -> dict:
    return {
        "total_goal_contributions": row.get("total_goal_contributions", 0) or 0,
        "total_shots": row.get("total_shots", 0) or 0,
        "total_assists": row.get("total_assists", 0) or 0,
        "total_market_value": row.get("total_market_value", 0) or 0,
        "total_interceptions": row.get("total_interceptions", 0) or 0,
        "count_DF": row.get("count_DF", 0) or 0,
        "squad_minutes_utilization": row.get("squad_minutes_utilization", 0) or 0,
        "count_GK": row.get("count_GK", 0) or 0,
        "avg_age": row.get("avg_age", 26) or 26,
    }


# ----------------------------------------------------------------------
# Monotonicity check: removing 50% of goals must NOT increase xP.
# ----------------------------------------------------------------------
def run_monotonicity_check(model: ClubSportingModel, cs: pd.DataFrame) -> bool:
    ok = True
    fails = 0
    for _, row in cs.iterrows():
        full = extract_squad_features(row.to_dict())
        reduced = dict(full)
        reduced["total_goal_contributions"] = full["total_goal_contributions"] * 0.5
        reduced["total_shots"] = full["total_shots"] * 0.5
        reduced["total_assists"] = full["total_assists"] * 0.5
        xp_full = model.predict_points(full)
        xp_reduced = model.predict_points(reduced)
        if xp_reduced > xp_full + 1e-6:
            fails += 1
            ok = False
    log.info("Monotonicity check (halving squad goals -> xP does NOT rise): %d / %d passed",
             len(cs) - fails, len(cs))
    return ok


# ----------------------------------------------------------------------
# LOOCV validation on the 34 valid transitions.
# Strategy: train capacity mappings on all OTHER team-seasons, then predict
# next-season points for the held-out t+1 squad features from its own squad
# state at t+1 (i.e. what we WOULD feed the scenario engine).
# ----------------------------------------------------------------------
def loocv_validation(cs: pd.DataFrame) -> dict:
    """Leave-one-club-season-out validation for next-season points.

    For each of the 34 valid transitions the base row (season s_cur) has
    points_next_season = the club's actual points in season s_next. The model
    is asked to predict next-season points from the club's SQUAD FEATURES at
    s_next (exactly what the scenario engine would feed after a departure).
    Training excludes that club's s_next observation (no leakage).
    """
    season_next = {188124: 189012, 189012: 189974}

    trans = cs[cs["points_next_season"].notna()].copy()
    log.info("LOOCV on %d valid transitions", len(trans))

    preds = []
    obs = []
    used = []
    for _, tr in trans.iterrows():
        s_cur = int(tr["season_id"])
        s_next = season_next.get(s_cur)
        club = int(tr["team_id"])
        # The squad feature row for next season (s_next) of this club
        next_row = cs[(cs["team_id"] == club) & (cs["season_id"] == s_next)]
        if next_row.empty:
            continue
        row0 = next_row.iloc[0]
        # Hold that next-season row OUT of training (prevent leakage)
        train = cs[~((cs["team_id"] == club) & (cs["season_id"] == s_next))]
        m = ClubSportingModel(s_next)
        m.fit(train)
        feats = extract_squad_features(row0.to_dict())
        xp = m.predict_points(feats)
        preds.append(xp)
        obs.append(tr["points_next_season"])
        used.append((club, s_cur, s_next))

    preds = np.array(preds)
    obs = np.array(obs)
    if len(preds) == 0:
        return {}
    resid = obs - preds
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss = np.sum((obs - obs.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / ss) if ss > 0 else 0.0
    corr = float(np.corrcoef(preds, obs)[0, 1]) if len(preds) > 1 else None
    log.info("LOOCV: n=%d RMSE=%.2f MAE=%.2f R2=%.3f corr=%.3f",
             len(preds), rmse, mae, r2, corr)
    return {
        "n": int(len(preds)),
        "rmse_points": round(rmse, 2),
        "mae_points": round(mae, 2),
        "r2_points": round(r2, 3),
        "corr": round(corr, 3) if corr is not None else None,
    }


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    cs = pd.read_csv(PROCESSED_DIR / "club_season_dataset.csv")
    log.info("Loaded club-season dataset: %s", cs.shape)

    # Train a model for the PRIMARY reference season (2024/25, season 189974).
    season_ref = 189974
    model = ClubSportingModel(season_ref)
    model.fit(cs)
    log.info("Fitted structural club model (reference season %s)", season_ref)

    # Monotonicity check
    ok = run_monotonicity_check(model, cs)

    # LOOCV validation
    cv = loocv_validation(cs)
    log.info("LOOCV results: %s", cv)

    # Feature importances (Ridge coefficients for attack, most interpretable
    # axis). Present standardised attack coefs.
    attack_coefs = dict(zip(ATTACK_FEATURES, model.attack_model.coef_))
    defence_coefs = dict(zip(DEFENCE_FEATURES, model.defence_model.coef_))
    log.info("Attack coefs (std units): %s", attack_coefs)
    log.info("Defence coefs (std units): %s", defence_coefs)

    # Persist
    with open(MODEL_DIR / "club_sporting_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "season_id": season_ref,
            "attack_features": ATTACK_FEATURES,
            "defence_features": DEFENCE_FEATURES,
            "attack_coefs": attack_coefs,
            "defence_coefs": defence_coefs,
            "r2_attack": getattr(model, "_attack_r2", None),
            "r2_defence": getattr(model, "_defence_r2", None),
        }, f)
    log.info("Saved model: %s", MODEL_DIR / "club_sporting_model.pkl")

    eval_out = {
        "model_strategy": "Structural Expected-Points (Poisson simulator) + "
                          "Ridge capacity mapping calibrated on pooled team-seasons",
        "reference_season_id": season_ref,
        "attack_r2": getattr(model, "_attack_r2", None),
        "defence_r2": getattr(model, "_defence_r2", None),
        "loocv": cv,
        "monotonicity_pass": ok,
        "attack_coefs_std_units": attack_coefs,
        "defence_coefs_std_units": defence_coefs,
        "n_transitions": int(cs["points_next_season"].notna().sum()),
    }
    with open(PROCESSED_DIR / "club_model_evaluation.json", "w") as f:
        json.dump(eval_out, f, indent=2)
    log.info("Saved evaluation: %s", PROCESSED_DIR / "club_model_evaluation.json")


if __name__ == "__main__":
    main()
