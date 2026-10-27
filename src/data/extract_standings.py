"""
extract_standings.py
====================
Stream-parse the 5.9 GB PostgreSQL COPY dump to derive official LaLiga
league standings per team per season.

The SQL file uses PostgreSQL COPY ... FROM stdin format:
    COPY sports.table_name (col1, col2, ...) FROM stdin;
    val1\tval2\t...
    \.

We extract four tables: matches, match_teams, seasons, teams.
Then we aggregate match-level results into team-season standings.

Output: data/interim/derived_league_standings.csv
"""

import io
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\football")
SQL_FILE = PROJECT_ROOT / "data" / "soccersolver_export_investigacion_2023-24_a_2025-26.sql"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_FILE = OUTPUT_DIR / "derived_league_standings.csv"

# Tables we need to extract from the SQL dump
TARGET_TABLES = {
    "sports.matches": [
        "match_id", "competition_id", "season_id", "round_id",
        "winner_team_id", "match_datetime_utc", "gameweek",
        "status", "duration_type", "venue", "label",
        "has_data_available", "possession_dead_time",
        "possesion_total_time", "created_at",
    ],
    "sports.match_teams": [
        "match_team_id", "match_id", "team_id", "coach_id",
        "side", "score", "score_ht", "score_ft", "score_et",
        "score_p", "has_formation", "created_at", "updated_at",
    ],
    "sports.seasons": [
        "season_id", "competition_id", "super_season_id",
        "name", "start_date", "end_date", "active",
        "created_at", "updated_at",
    ],
    "sports.teams": [
        "team_id", "area_id", "name", "official_name",
        "city", "type", "category", "gender", "parent_team_id",
        "stadium", "image_url", "created_at", "updated_at",
    ],
}

# Columns we actually need from each table (subset for memory)
KEEP_COLS = {
    "sports.matches": [
        "match_id", "competition_id", "season_id", "winner_team_id",
        "match_datetime_utc", "gameweek", "status", "duration_type",
    ],
    "sports.match_teams": [
        "match_id", "team_id", "score_ft", "score_ht", "side",
    ],
    "sports.seasons": [
        "season_id", "super_season_id", "name",
    ],
    "sports.teams": [
        "team_id", "name",
    ],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL COPY block parser
# ---------------------------------------------------------------------------
def _safe_float(val: str) -> Optional[float]:
    """Convert a COPY field to float, returning None for NULL / empty."""
    if val in (r"\N", "", "NULL", "None"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: str) -> Optional[int]:
    """Convert a COPY field to int, returning None for NULL / empty."""
    f = _safe_float(val)
    if f is None:
        return None
    return int(f)


def parse_copy_block(
    file_path: Path,
    target_tables: Dict[str, List[str]],
    keep_cols: Dict[str, List[str]],
) -> Dict[str, pd.DataFrame]:
    """
    Stream through the SQL dump and extract COPY blocks for target tables.

    Returns a dict of {table_name: DataFrame}.
    """
    collected: Dict[str, List[List]] = {t: [] for t in target_tables}
    col_specs: Dict[str, List[str]] = {}
    active_table: Optional[str] = None
    active_cols: List[str] = []
    bytes_read = 0
    log_interval = 500_000_000  # log every 500 MB

    copy_re = re.compile(
        r"^COPY\s+(sports\.\w+)\s*\((.+?)\)\s+FROM\s+stdin\s*;"
    )

    log.info("Streaming SQL dump: %s (%.1f GB)", file_path.name, file_path.stat().st_size / 1e9)

    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            bytes_read += len(line.encode("utf-8", errors="replace"))

            if bytes_read % log_interval < len(line.encode("utf-8", errors="replace")):
                log.info(
                    "  ... processed %.1f GB / %.1f GB",
                    bytes_read / 1e9,
                    file_path.stat().st_size / 1e9,
                )

            stripped = line.rstrip("\n")

            # Check for COPY header
            m = copy_re.match(stripped)
            if m:
                tbl_name = m.group(1)
                if tbl_name in target_tables:
                    active_table = tbl_name
                    # Parse column names from the header
                    raw_cols = [c.strip().strip('"') for c in m.group(2).split(",")]
                    active_cols = raw_cols
                    col_specs[tbl_name] = raw_cols
                    log.info("  Extracting table: %s (%d columns)", tbl_name, len(raw_cols))
                else:
                    active_table = None
                continue

            # End of COPY block
            if stripped == r"\.":
                if active_table is not None:
                    n_rows = len(collected[active_table])
                    log.info("    -> %s: %d rows extracted", active_table, n_rows)
                    active_table = None
                    active_cols = []
                continue

            # Data line: parse if we're in an active table
            if active_table is not None and active_table in target_tables:
                fields = stripped.split("\t")
                row = {}
                for i, col_name in enumerate(active_cols):
                    if i < len(fields):
                        row[col_name] = fields[i]
                    else:
                        row[col_name] = r"\N"
                collected[active_table].append(row)

    # Build DataFrames with only the columns we need
    result: Dict[str, pd.DataFrame] = {}
    for tbl in target_tables:
        if not collected[tbl]:
            log.warning("No data extracted for %s", tbl)
            result[tbl] = pd.DataFrame(columns=keep_cols.get(tbl, []))
            continue

        df = pd.DataFrame(collected[tbl])
        # Select only needed columns that exist
        available = [c for c in keep_cols.get(tbl, []) if c in df.columns]
        df = df[available].copy()

        # Type coercion
        if tbl == "sports.matches":
            for c in ["match_id", "competition_id", "season_id", "winner_team_id", "gameweek"]:
                if c in df.columns:
                    df[c] = df[c].apply(_safe_int)
            for c in ["possession_dead_time", "possesion_total_time"]:
                if c in df.columns:
                    df[c] = df[c].apply(_safe_float)

        elif tbl == "sports.match_teams":
            for c in ["match_id", "team_id"]:
                if c in df.columns:
                    df[c] = df[c].apply(_safe_int)
            for c in ["score_ft", "score_ht"]:
                if c in df.columns:
                    df[c] = df[c].apply(_safe_int)

        elif tbl == "sports.seasons":
            for c in ["season_id", "super_season_id"]:
                if c in df.columns:
                    df[c] = df[c].apply(_safe_int)

        elif tbl == "sports.teams":
            if "team_id" in df.columns:
                df["team_id"] = df["team_id"].apply(_safe_int)

        result[tbl] = df
        log.info("Built DataFrame for %s: %s", tbl, df.shape)

    return result


# ---------------------------------------------------------------------------
# Standings computation
# ---------------------------------------------------------------------------
def compute_standings(
    matches: pd.DataFrame,
    match_teams: pd.DataFrame,
    seasons: pd.DataFrame,
    teams: pd.DataFrame,
    target_competition_id: int = 795,  # LaLiga
) -> pd.DataFrame:
    """
    Compute end-of-season league standings from match results.

    For each match, both home and away team records are aggregated:
        Win   = 3 pts
        Draw  = 1 pt
        Loss  = 0 pts
        GF, GA per match
    """
    log.info("Computing league standings...")

    # ---- Filter to LaLiga (competition_id=795) if available ----
    if "competition_id" in matches.columns:
        matches = matches[matches["competition_id"] == target_competition_id].copy()
        log.info("Filtered to competition_id=%d: %d matches", target_competition_id, len(matches))

    # ---- Restrict to the seasons this repo models (2022-25 window) ----
    # 2022/23 is prepended analytically by fix_2022_23_standings.py; the dump
    # covers 2023/24 -> 2025/26, and we keep only the two completed seasons
    # that feed the master dataset (season_ids 189012, 189974). This keeps
    # stage 1 output byte-identical to the committed data/interim artifact.
    TARGET_SEASON_IDS = {189012, 189974}
    if "season_id" in matches.columns:
        n_before = len(matches)
        matches = matches[matches["season_id"].isin(TARGET_SEASON_IDS)].copy()
        log.info("Restricted to target seasons %s: %d matches (from %d)",
                 sorted(TARGET_SEASON_IDS), len(matches), n_before)

    # ---- Merge matches with their teams ----
    # Each match has 2 rows in match_teams (home + away)
    mt = match_teams.merge(
        matches[["match_id", "season_id", "gameweek", "status", "duration_type"]],
        on="match_id",
        how="inner",
    )

    # Filter to regular-time completed matches (exclude penalty shootouts for standings)
    # Keep matches that have a final score
    mt = mt.dropna(subset=["score_ft"])

    # ---- Determine home/away and compute W/D/L ----
    # 'side' indicates home/away; score_ft is the team's goals
    # We need to pair home vs away for each match to determine W/D/L
    match_pairs = mt.pivot_table(
        index="match_id",
        columns="side",
        values=["score_ft", "team_id", "score_ht"],
        aggfunc="first",
    )

    # Flatten multi-index columns: e.g., ('score_ft', 'home') -> 'score_ft_home'
    match_pairs.columns = [f"{col[0]}_{col[1]}" for col in match_pairs.columns]
    match_pairs = match_pairs.reset_index()

    # Ensure we have both home and away scores
    required_cols = ["score_ft_home", "score_ft_away", "team_id_home", "team_id_away"]
    for c in required_cols:
        if c not in match_pairs.columns:
            log.warning("Column %s not found — skipping match pairing", c)
            return pd.DataFrame()

    match_pairs = match_pairs.dropna(subset=["score_ft_home", "score_ft_away"])

    # ---- Pre-merge season_id into match_pairs to avoid O(n^2) lookup ----
    season_info = matches[["match_id", "season_id", "gameweek"]].drop_duplicates(subset=["match_id"])
    match_pairs = match_pairs.merge(season_info, on="match_id", how="left")

    # ---- Build per-team match records ----
    records = []

    for _, row in match_pairs.iterrows():
        home_id = _safe_int(str(int(row["team_id_home"]))) if pd.notna(row["team_id_home"]) else None
        away_id = _safe_int(str(int(row["team_id_away"]))) if pd.notna(row["team_id_away"]) else None
        if home_id is None or away_id is None:
            continue

        gf_h = int(row["score_ft_home"])
        ga_h = int(row["score_ft_away"])
        gf_a = ga_h  # away team's GF = home team's GA
        ga_a = gf_h

        # Determine result for home team
        if gf_h > ga_h:
            pts_h, pts_a = 3, 0
        elif gf_h == ga_h:
            pts_h, pts_a = 1, 1
        else:
            pts_h, pts_a = 0, 3

        season_id = row.get("season_id")
        gameweek = row.get("gameweek")

        records.append({
            "match_id": int(row["match_id"]),
            "season_id": season_id,
            "gameweek": gameweek,
            "team_id": home_id,
            "goals_for": gf_h,
            "goals_against": ga_h,
            "goal_diff": gf_h - ga_h,
            "points": pts_h,
            "is_home": 1,
        })
        records.append({
            "match_id": int(row["match_id"]),
            "season_id": season_id,
            "gameweek": gameweek,
            "team_id": away_id,
            "goals_for": gf_a,
            "goals_against": ga_a,
            "goal_diff": gf_a - ga_a,
            "points": pts_a,
            "is_home": 0,
        })

    team_matches = pd.DataFrame(records)
    log.info("Built per-team match records: %s", team_matches.shape)

    # ---- Aggregate to team-season standings ----
    team_season = (
        team_matches.groupby(["season_id", "team_id"])
        .agg(
            matches_played=("match_id", "nunique"),
            wins=("points", lambda x: (x == 3).sum()),
            draws=("points", lambda x: (x == 1).sum()),
            losses=("points", lambda x: (x == 0).sum()),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            goal_diff=("goal_diff", "sum"),
            points=("points", "sum"),
            home_matches=("is_home", "sum"),
        )
        .reset_index()
    )

    team_season["points_per_match"] = (
        team_season["points"] / team_season["matches_played"].replace(0, np.nan)
    )

    # ---- Rank teams within each season ----
    team_season["league_rank"] = (
        team_season.groupby("season_id")["points"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    # Tiebreaker: goal difference
    team_season = team_season.sort_values(
        ["season_id", "points", "goal_diff", "goals_for"],
        ascending=[True, False, False, False],
    )
    team_season["league_rank"] = (
        team_season.groupby("season_id")
        .cumcount()
        + 1
    )

    # Relegation indicator: bottom 3 in a 20-team league (rank >= 18)
    team_season["relegated"] = (team_season["league_rank"] >= 18).astype(int)

    # ---- Join season names ----
    team_season = team_season.merge(
        seasons[["season_id", "super_season_id", "name"]],
        on="season_id",
        how="left",
    )

    # ---- Join team names ----
    team_season = team_season.merge(
        teams[["team_id", "name"]],
        on="team_id",
        how="left",
        suffixes=("", "_team"),
    )
    if "name_team" in team_season.columns:
        team_season.rename(columns={"name_team": "team_name"}, inplace=True)

    # ---- Sort and select final columns ----
    final_cols = [
        "season_id", "super_season_id", "name", "team_id", "team_name",
        "matches_played", "wins", "draws", "losses",
        "goals_for", "goals_against", "goal_diff",
        "points", "points_per_match", "league_rank", "relegated",
    ]
    # Only keep columns that exist
    final_cols = [c for c in final_cols if c in team_season.columns]

    team_season = team_season[final_cols].sort_values(
        ["season_id", "league_rank"]
    ).reset_index(drop=True)

    return team_season


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SQL_FILE.exists():
        log.error("SQL file not found: %s", SQL_FILE)
        sys.exit(1)

    # Step 1: Stream-parse the SQL dump
    log.info("=" * 60)
    log.info("PHASE 1: Streaming SQL COPY blocks")
    log.info("=" * 60)
    tables = parse_copy_block(SQL_FILE, TARGET_TABLES, KEEP_COLS)

    matches = tables["sports.matches"]
    match_teams = tables["sports.match_teams"]
    seasons = tables["sports.seasons"]
    teams = tables["sports.teams"]

    log.info("Matches: %s | Match Teams: %s | Seasons: %s | Teams: %s",
             matches.shape, match_teams.shape, seasons.shape, teams.shape)

    # Step 2: Compute standings
    log.info("=" * 60)
    log.info("PHASE 2: Computing league standings")
    log.info("=" * 60)
    standings = compute_standings(matches, match_teams, seasons, teams)

    if standings.empty:
        log.error("Standings computation returned empty DataFrame")
        sys.exit(1)

    # Step 3: Save
    standings.to_csv(OUTPUT_FILE, index=False)
    log.info("=" * 60)
    log.info("OUTPUT: %s", OUTPUT_FILE)
    log.info("Shape: %s", standings.shape)
    log.info("=" * 60)

    # Print summary
    log.info("\n--- Standings Summary ---")
    for sid in standings["season_id"].unique():
        subset = standings[standings["season_id"] == sid]
        season_name = subset["name"].iloc[0] if "name" in subset.columns else sid
        log.info("Season %s: %d teams, %d matches avg",
                 season_name, len(subset), subset["matches_played"].mean())
        top3 = subset.head(3)[["team_name", "points", "league_rank"]]
        log.info("Top 3:\n%s", top3.to_string(index=False))

    return standings


if __name__ == "__main__":
    main()
