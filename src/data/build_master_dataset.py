"""
build_master_dataset.py
=======================
8-step join pipeline to create a unified player-season master dataset.

Join chain:
  1. player_season_stats.parquet (bridge: player_id, team_id, super_season_id)
  2. seasons.parquet (super_season_id -> season_id, name)
  3. player_careers.parquet (player_id + team_id + season_id -> shirt_number)
  4. players.parquet (player_id -> demographics)
  5. teams.parquet (team_id -> name)
  6. player_season_stats_{total,per90,percent,average}.parquet (stats)
  7. player_valuations.parquet (market value per season)
  8. derived_league_standings.csv (team standings per season)

Output: data/interim/master_dataset.csv
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(r"C:\football")
PARQUET_DIR = PROJECT_ROOT / "data" / "laliga_2022_2025"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_FILE = INTERIM_DIR / "master_dataset.csv"


def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(PARQUET_DIR / f"{name}.parquet")


def extract_primary_position(positions_json: str) -> str:
    """Extract the position played for the most minutes."""
    try:
        positions = json.loads(positions_json)
        if positions:
            return positions[0].get("position", {}).get("code", "unknown")
    except (json.JSONDecodeError, TypeError):
        pass
    return "unknown"


def build_master_dataset() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("STEP 1: Load bridge table (player_season_stats)")
    log.info("=" * 60)
    pss = load_parquet("player_season_stats")
    log.info("  Shape: %s", pss.shape)
    log.info("  super_season_ids: %s", sorted(pss["super_season_id"].unique()))

    log.info("=" * 60)
    log.info("STEP 2: Join seasons -> map super_season_id to season_id")
    log.info("=" * 60)
    seasons = load_parquet("seasons")
    log.info("  Seasons:\n%s", seasons[["season_id", "super_season_id", "name"]].to_string(index=False))

    season_map = seasons[["super_season_id", "season_id", "name"]].rename(
        columns={"name": "season_name"}
    )
    pss = pss.merge(season_map, on="super_season_id", how="left")
    log.info("  After join: %s", pss.shape)
    log.info("  Null season_ids: %d", pss["season_id"].isna().sum())

    log.info("=" * 60)
    log.info("STEP 3: Join player_careers -> shirt_number")
    log.info("=" * 60)
    careers = load_parquet("player_careers")
    careers_dedup = careers.drop_duplicates(subset=["player_id", "team_id", "season_id"])
    pss = pss.merge(
        careers_dedup[["player_id", "team_id", "season_id", "shirt_number"]],
        on=["player_id", "team_id", "season_id"],
        how="left",
    )
    log.info("  After join: %s", pss.shape)
    log.info("  Null shirt_numbers: %d", pss["shirt_number"].isna().sum())

    log.info("=" * 60)
    log.info("STEP 4: Join players -> demographics")
    log.info("=" * 60)
    players = load_parquet("players")
    pss = pss.merge(
        players[["player_id", "short_name", "first_name", "last_name",
                  "birth_date", "role_code", "role_name", "foot",
                  "height_cm", "weight_kg", "status"]],
        on="player_id",
        how="left",
    )
    log.info("  After join: %s", pss.shape)

    if "birth_date" in pss.columns:
        pss["birth_date"] = pd.to_datetime(pss["birth_date"], errors="coerce")
        season_end = {2023: "2023-06-30", 2024: "2024-06-30", 2025: "2025-06-30"}
        pss["season_end_date"] = pss["super_season_id"].map(season_end).apply(pd.to_datetime)
        pss["age_at_season_end"] = (
            (pss["season_end_date"] - pss["birth_date"]).dt.days / 365.25
        ).round(1)

    log.info("=" * 60)
    log.info("STEP 5: Join teams -> team name")
    log.info("=" * 60)
    teams = load_parquet("teams")
    pss = pss.merge(
        teams[["team_id", "name"]].rename(columns={"name": "team_name"}),
        on="team_id",
        how="left",
    )
    log.info("  After join: %s", pss.shape)

    log.info("=" * 60)
    log.info("STEP 6: Join stat tables (total, per90, percent, average)")
    log.info("=" * 60)
    for stat_suffix in ["total", "per90", "percent", "average"]:
        stat_df = load_parquet(f"player_season_stats_{stat_suffix}")
        stat_cols = [c for c in stat_df.columns if c != "player_season_stats_id"]
        prefix = f"stat_{stat_suffix}_"
        rename_map = {c: f"{prefix}{c}" for c in stat_cols}
        stat_df = stat_df.rename(columns=rename_map)
        pss = pss.merge(stat_df, on="player_season_stats_id", how="left")
        log.info("  + %s: %d new cols, total shape %s",
                 stat_suffix, len(stat_cols), pss.shape)

    if "positions" in pss.columns:
        pss["primary_position"] = pss["positions"].apply(extract_primary_position)
        log.info("  Position distribution:\n%s",
                 pss["primary_position"].value_counts().head(10).to_string())

    log.info("=" * 60)
    log.info("STEP 7: Join valuations -> market value")
    log.info("=" * 60)
    valuations = load_parquet("player_valuations")
    valuations["valuation_date"] = pd.to_datetime(valuations["valuation_date"], errors="coerce")

    season_dates = seasons[["season_id", "start_date", "end_date"]].copy()
    season_dates["start_date"] = pd.to_datetime(season_dates["start_date"])
    season_dates["end_date"] = pd.to_datetime(season_dates["end_date"])

    val_with_team = valuations.dropna(subset=["team_id"]).copy()
    val_with_team["team_id"] = val_with_team["team_id"].astype(int)

    val_merged = val_with_team.merge(season_dates, how="cross")
    in_season = (
        (val_merged["valuation_date"] >= val_merged["start_date"])
        & (val_merged["valuation_date"] <= val_merged["end_date"])
    )
    val_in_season = val_merged[in_season].copy()
    val_in_season = val_in_season.sort_values("valuation_date")
    val_latest = (
        val_in_season
        .groupby(["player_id", "team_id", "season_id"])
        .last()
        .reset_index()
        [["player_id", "team_id", "season_id", "market_value"]]
        .rename(columns={"market_value": "market_value_eur"})
    )

    val_no_team = valuations[valuations["team_id"].isna()].copy()
    val_no_team_merged = val_no_team.merge(season_dates, how="cross")
    in_season_nt = (
        (val_no_team_merged["valuation_date"] >= val_no_team_merged["start_date"])
        & (val_no_team_merged["valuation_date"] <= val_no_team_merged["end_date"])
    )
    val_no_team_in = val_no_team_merged[in_season_nt].sort_values("valuation_date")
    val_no_team_latest = (
        val_no_team_in
        .groupby(["player_id", "season_id"])
        .last()
        .reset_index()
        [["player_id", "season_id", "market_value"]]
        .rename(columns={"market_value": "market_value_eur"})
    )

    pss = pss.merge(val_latest, on=["player_id", "team_id", "season_id"], how="left")
    missing_val = pss["market_value_eur"].isna()
    if missing_val.any():
        fallback = pss.loc[missing_val, ["player_id", "season_id"]].merge(
            val_no_team_latest,
            on=["player_id", "season_id"],
            how="left",
        )
        pss.loc[missing_val, "market_value_eur"] = fallback["market_value_eur"].values

    log.info("  Valuations joined: %d non-null out of %d",
             pss["market_value_eur"].notna().sum(), len(pss))

    log.info("=" * 60)
    log.info("STEP 8: Join derived league standings")
    log.info("=" * 60)
    standings = pd.read_csv(INTERIM_DIR / "derived_league_standings.csv")
    standings_join = standings[[
        "season_id", "team_id", "matches_played", "wins", "draws", "losses",
        "goals_for", "goals_against", "goal_diff", "points", "points_per_match",
        "league_rank", "relegated",
    ]].rename(columns={
        "matches_played": "team_matches",
        "wins": "team_wins",
        "draws": "team_draws",
        "losses": "team_losses",
        "goals_for": "team_gf",
        "goals_against": "team_ga",
        "goal_diff": "team_gd",
        "points": "team_points",
        "points_per_match": "team_ppm",
        "league_rank": "team_league_rank",
        "relegated": "team_relegated",
    })
    pss = pss.merge(standings_join, on=["team_id", "season_id"], how="left")
    log.info("  After join: %s", pss.shape)
    log.info("  Null team_league_rank: %d", pss["team_league_rank"].isna().sum())

    # ----------------------------------------------------------------
    # Final cleanup
    # ----------------------------------------------------------------
    drop_cols = ["season_end_date", "positions", "status"]
    for c in drop_cols:
        if c in pss.columns:
            pss.drop(columns=[c], inplace=True)

    pss = pss.sort_values(["season_id", "team_id", "player_id"]).reset_index(drop=True)

    log.info("=" * 60)
    log.info("FINAL MASTER DATASET")
    log.info("=" * 60)
    log.info("Shape: %s", pss.shape)
    log.info("Seasons: %s", sorted(pss["season_id"].unique()))
    log.info("Players: %d unique", pss["player_id"].nunique())
    log.info("Teams: %d unique", pss["team_id"].nunique())
    log.info("Market value coverage: %.1f%%", pss["market_value_eur"].notna().mean() * 100)
    log.info("team_league_rank coverage: %.1f%%", pss["team_league_rank"].notna().mean() * 100)

    return pss


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df = build_master_dataset()
    df.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved: %s (%d rows x %d cols)", OUTPUT_FILE, *df.shape)

    summary = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "seasons": sorted(df["season_id"].unique().tolist()),
        "n_players": int(df["player_id"].nunique()),
        "n_teams": int(df["team_id"].nunique()),
        "market_value_coverage": float(df["market_value_eur"].notna().mean()),
    }
    summary_file = INTERIM_DIR / "master_dataset_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary: %s", summary_file)


if __name__ == "__main__":
    main()
