"""
build_club_season_dataset.py
============================
Aggregates player-season master data to club-season level.

Feature families:
  A. Squad financials (market value aggregates)
  B. Squad composition (position counts, age profile)
  C. Squad performance (goals, assists, minutes)
  D. Goal concentration (HHI of goal contributions)
  E. Departure features (players who left, value lost)
  F. Lagged targets (next-season points/rank)
  G. Observed outcome labels (Stage 17a promoted/relegated flags; the old
     position-based relegation proxy is retained as team_relegated_proxy)

Output: data/processed/club_season_dataset.csv
"""

import logging
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(r"C:\football")
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "club_season_dataset.csv"

POSITION_GROUP_MAP = {
    "gk": "GK",
    "lcb": "DF", "rcb": "DF", "cb": "DF", "lb": "DF", "rb": "DF", "lwb": "DF", "rwb": "DF",
    "lcb3": "DF", "rcb3": "DF", "lb5": "DF", "rb5": "DF", "cb3": "DF",
    "lcmf": "MF", "rcmf": "MF", "dmf": "MF", "cmf": "MF", "lamf": "MF", "ramf": "MF",
    "lcmf3": "MF", "rcmf3": "MF", "ldmf": "MF", "rdmf": "MF", "amf": "MF",
    "lw": "FW", "rw": "FW", "lwf": "FW", "rwf": "FW", "cf": "FW", "ss": "FW",
}


def map_position_group(pos: str) -> str:
    """Map a primary_position code to a broad group (GK/DF/MF/FW).
    Falls back to a prefix heuristic for unlisted album-codes."""
    if pos in POSITION_GROUP_MAP:
        return POSITION_GROUP_MAP[pos]
    pos_low = pos.lower()
    # Heuristic: strip trailing digits (e.g. 'rcmf3' -> 'rcmf')
    base = pos_low.rstrip("0123456789")
    return POSITION_GROUP_MAP.get(base, "UNK")


def load_master() -> pd.DataFrame:
    df = pd.read_csv(INTERIM_DIR / "master_dataset.csv")
    df["position_group"] = df["primary_position"].apply(map_position_group)
    return df


def compute_gini(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def compute_hhi(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    total = arr.sum()
    if total == 0:
        return 0.0
    shares = arr / total
    return float(np.sum(shares ** 2))


def aggregate_squad(df_group):
    row = {}
    n = len(df_group)

    row["squad_size"] = n

    # --- A. Squad financials ---
    mv = df_group["market_value_eur"].dropna()
    row["total_market_value"] = float(mv.sum()) if len(mv) > 0 else np.nan
    row["avg_market_value"] = float(mv.mean()) if len(mv) > 0 else np.nan
    row["median_market_value"] = float(mv.median()) if len(mv) > 0 else np.nan
    row["max_market_value"] = float(mv.max()) if len(mv) > 0 else np.nan
    row["market_value_gini"] = compute_gini(mv.values) if len(mv) > 0 else np.nan
    row["n_valued_players"] = len(mv)

    # --- B. Squad composition ---
    for grp in ["GK", "DF", "MF", "FW"]:
        row[f"count_{grp}"] = int((df_group["position_group"] == grp).sum())
    row["position_diversity"] = df_group["position_group"].nunique()

    # Age profile
    ages = df_group["age_at_season_end"].dropna()
    row["avg_age"] = float(ages.mean()) if len(ages) > 0 else np.nan
    row["median_age"] = float(ages.median()) if len(ages) > 0 else np.nan
    row["max_age"] = float(ages.max()) if len(ages) > 0 else np.nan
    row["min_age"] = float(ages.min()) if len(ages) > 0 else np.nan
    row["u23_count"] = int((ages < 23).sum()) if len(ages) > 0 else 0
    row["over30_count"] = int((ages > 30).sum()) if len(ages) > 0 else 0

    # --- C. Squad performance ---
    for stat, label in [
        ("stat_total_goals", "total_goals"),
        ("stat_total_assists", "total_assists"),
        ("stat_total_minutes_played", "total_minutes"),
        ("stat_total_yellow_cards", "total_yellows"),
        ("stat_total_red_cards", "total_reds"),
        ("stat_total_shots", "total_shots"),
        ("stat_total_key_passes", "total_key_passes"),
        ("stat_total_interceptions", "total_interceptions"),
    ]:
        if stat in df_group.columns:
            row[label] = float(df_group[stat].sum())
        else:
            row[label] = np.nan

    # Goal contributions (goals + assists)
    goals = df_group.get("stat_total_goals", pd.Series(dtype=float))
    assists = df_group.get("stat_total_assists", pd.Series(dtype=float))
    gc = goals.fillna(0) + assists.fillna(0)
    row["total_goal_contributions"] = float(gc.sum())

    # Minutes utilization
    max_minutes = 90 * 38
    if "stat_total_minutes_played" in df_group.columns:
        total_mins = df_group["stat_total_minutes_played"].sum()
        row["squad_minutes_utilization"] = float(total_mins / (max_minutes * n)) if n > 0 else 0.0
    else:
        row["squad_minutes_utilization"] = np.nan

    # --- D. Goal concentration (HHI) ---
    row["goal_concentration_hhi"] = compute_hhi(gc.values)

    # Top scorer share
    if gc.sum() > 0:
        row["top_scorer_share"] = float(gc.max() / gc.sum())
    else:
        row["top_scorer_share"] = np.nan

    # --- E. Card discipline ---
    row["yellow_card_rate"] = row["total_yellows"] / (row["total_minutes"] / 90) if row["total_minutes"] > 0 else 0.0
    row["red_card_rate"] = row["total_reds"] / (row["total_minutes"] / 90) if row["total_minutes"] > 0 else 0.0

    # --- F. Squad value tiers ---
    if len(mv) > 0:
        row["n_above_10m"] = int((mv > 10_000_000).sum())
        row["n_above_20m"] = int((mv > 20_000_000).sum())
        row["n_above_50m"] = int((mv > 50_000_000).sum())
        top5_val = mv.nlargest(5).sum() if len(mv) >= 5 else mv.sum()
        row["top5_concentration"] = float(top5_val / mv.sum()) if mv.sum() > 0 else np.nan
    else:
        row["n_above_10m"] = 0
        row["n_above_20m"] = 0
        row["n_above_50m"] = 0
        row["top5_concentration"] = np.nan

    return pd.Series(row)


def compute_departures(df, seasons_sorted):
    """For each club-season pair, compute how many players left and their total value."""
    departures = {}
    for i in range(len(seasons_sorted) - 1):
        s_curr = seasons_sorted[i]
        s_next = seasons_sorted[i + 1]

        curr_set = df[df["season_id"] == s_curr].set_index(["team_id", "player_id"])["market_value_eur"]
        next_set = df[df["season_id"] == s_next].set_index(["team_id", "player_id"])["market_value_eur"]

        common_keys = curr_set.index.intersection(next_set.index)
        departed_keys = curr_set.index.difference(next_set.index)

        for team_id in df[df["season_id"] == s_curr]["team_id"].unique():
            key = (team_id, s_curr)
            team_departed = [k for k in departed_keys if k[0] == team_id]
            n_departed = len(team_departed)
            val_departed = sum(
                curr_set.loc[k] for k in team_departed if pd.notna(curr_set.loc[k])
            )
            n_remaining = len([k for k in common_keys if k[0] == team_id])
            departures[key] = {
                "players_departed": n_departed,
                "departures_value": float(val_departed),
                "players_remaining": n_remaining,
                "departure_rate": n_departed / (n_departed + n_remaining) if (n_departed + n_remaining) > 0 else 0.0,
            }
    return departures


# Stage 17a season_code -> internal season_id lookup. Internal master spans
# 2022/23 (188124), 2023/24 (189012), 2024/25 (189974).
SEASON_CODE_TO_ID = {2223: 188124, 2324: 189012, 2425: 189974}

# football-data.co.uk team_name_norm -> internal (master) team_name spelling.
# fd-co.uk abbreviates/omits common prefixes, so we normalize both sides to an
# accent-free lower key and only need aliases for the residual mismatches.
FD_TEAM_NAME_ALIASES = {
    # SP1 primary spellings after de-accenting
    "alaves": ["deportivo alaves"],
    "atletico madrid": ["atletico madrid"],
    "athletic club": ["athletic club"],
    "barcelona": ["barcelona"],
    "betis": ["real betis"],
    "cadiz": ["cadiz"],
    "celta": ["celta de vigo"],
    "elche": ["elche"],
    "espanol": ["espanyol"],
    "getafe": ["getafe"],
    "girona": ["girona"],
    "granada": ["granada"],
    "las palmas": ["las palmas"],
    "leganes": ["leganes"],
    "mallorca": ["mallorca"],
    "osasuna": ["osasuna"],
    "real madrid": ["real madrid"],
    "sevilla": ["sevilla"],
    "sociedad": ["real sociedad"],
    "valencia": ["valencia"],
    "valladolid": ["real valladolid"],
    "vallecano": ["rayo vallecano"],
    "villarreal": ["villarreal"],
}


def normalize_team_name(name: str) -> str:
    """Lower and de-accent a team name to a joinable key key."""
    if not isinstance(name, str):
        return ""
    name = "".join(
        ch for ch in unicodedata.normalize("NFD", name) if unicodedata.category(ch) != "Mn"
    )
    return " ".join(name.lower().split())


def canonical_to_fd_key(team_name: str) -> str:
    """Map an internal (master) team name to the fd-co.uk team_name_norm key."""
    norm = normalize_team_name(team_name)
    for fd_key, internal_forms in FD_TEAM_NAME_ALIASES.items():
        if norm in internal_forms:
            return fd_key
    # A non-La-Liga spelling (e.g. a promoted side) simply normalizes.
    return norm


def join_observed_outcome_flags(agg: pd.DataFrame) -> pd.DataFrame:
    """Join Stage 17a OBSERVED promoted/relegated flags onto club-seasons.

    Replaces the old position-based relegation proxy (kept as
    team_relegated_proxy) with the observed flag. Rows in the terminal season
    (flag_method = terminal_season_unobserved) keep the proxy label, which is
    recorded in team_relegated_flag_method for provenance. Unmatched club names
    are logged, never silently dropped.
    """
    src = INTERIM_DIR / "derived_league_standings_multi.csv"
    if not src.exists():
        agg["team_relegated_proxy"] = agg["team_relegated"]
        agg["team_relegated"] = agg["team_relegated_proxy"]
        agg["team_relegated_flag_method"] = "proxy_no_stage17a"
        agg["team_promoted"] = 0
        agg["team_promoted_flag_method"] = "proxy_no_stage17a"
        log.warning("WARNING: %s missing; observed labels unavailable, proxy retained.", src)
        return agg

    s17 = pd.read_csv(src)
    s17 = s17[s17["division_code"].isin(["SP1", "SP2"])].copy()
    s17["season_id"] = s17["season_code"].map(SEASON_CODE_TO_ID)
    s17 = s17.dropna(subset=["season_id"])

    # Keep old proxy under a _proxy suffix before overwriting.
    if "team_relegated" in agg.columns:
        agg["team_relegated_proxy"] = agg["team_relegated"]
    else:
        agg["team_relegated_proxy"] = np.nan

    agg["_join_key"] = agg["team_name"].map(canonical_to_fd_key)
    s17["_join_key"] = s17["team_name_norm"].map(normalize_team_name)

    matched = agg[["_join_key", "season_id"]].merge(
        s17[["_join_key", "season_id", "promoted", "relegated", "flag_method"]],
        on=["_join_key", "season_id"],
        how="left",
        validate="one_to_one",
    )
    obs_rel = matched["relegated"].astype(float)  # 1.0 / 0.0 / NaN
    obs_prom = matched["promoted"].astype(float)
    agg["team_relegated"] = np.where(
        obs_rel.isna(), agg["team_relegated_proxy"], obs_rel
    )
    agg["team_promoted"] = np.where(obs_prom.isna(), 0, obs_prom)
    agg["team_relegated_flag_method"] = matched["flag_method"].fillna("unmatched_proxy")
    agg["team_promoted_flag_method"] = matched["flag_method"].fillna("unmatched_proxy")

    unmatched = agg[matched["_join_key"].isna() & matched["flag_method"].isna()][
        ["team_name", "season_id"]
    ]
    if not unmatched.empty:
        log.warning("WARNING: %d club-seasons unmatched by Stage 17a name:", len(unmatched))
        for _, r in unmatched.iterrows():
            log.warning("  - %s (season_id=%d)", r["team_name"], r["season_id"])
    else:
        log.info("All club-seasons matched to Stage 17a SP1/SP2 flags by name.")

    # terminal-season rows keep proxy: re-apply the label source for clarity
    term = matched["flag_method"].eq("terminal_season_unobserved")
    agg.loc[term, "team_relegated"] = agg.loc[term, "team_relegated_proxy"]

    # comparison: proxy (rank>=18) vs observed, restricted to observed flags
    # (Stage 17a writes terminal rows as False, not NaN, so filter by method)
    obs_avail = matched["flag_method"].eq("observed_next_season")
    comp = agg.loc[obs_avail, ["team_name", "season_id", "team_relegated_proxy", "team_relegated"]]
    obs_rel_comp = obs_rel[obs_avail].astype(int).reset_index(drop=True)
    comp["observed"] = obs_rel_comp.values
    comp["diff"] = comp["team_relegated_proxy"].astype(int) != comp["observed"]
    log.info("=" * 60)
    log.info("OBSERVED vs PROXY relegation labels (Stage 17a)")
    log.info("=" * 60)
    log.info("  Observed flags available (non-terminal): %d club-seasons", int(obs_avail.sum()))
    log.info("  proxy != observed: %d club-seasons", int(comp["diff"].sum()))
    if comp["diff"].any():
        for _, r in comp[comp["diff"]].iterrows():
            log.info("    %-20s season=%d proxy=%d observed=%d",
                     r["team_name"], r["season_id"], r["team_relegated_proxy"], r["team_relegated"])
    n_term = int(term.sum())
    if n_term:
        log.info("  %d club-seasons (2024/25 terminal) have no next-season data;",
                 n_term)
        log.info("  proxy label retained (flag_method=%s)", "'terminal_season_unobserved'")

    agg["_join_key"] = matched["_join_key"]

    agg.drop(columns=["_join_key"], inplace=True)
    return agg


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading master dataset...")
    df = load_master()
    log.info("  Shape: %s", df.shape)

    seasons_sorted = sorted(df["season_id"].unique())
    log.info("  Seasons: %s", seasons_sorted)

    # --- Aggregate by (team_id, season_id) ---
    log.info("Aggregating squad features by club-season...")
    agg = df.groupby(["team_id", "season_id"]).apply(aggregate_squad).reset_index()

    # Add team_name
    team_names = df.drop_duplicates("team_id")[["team_id", "team_name"]]
    agg = agg.merge(team_names, on="team_id", how="left")

    # Add league standing features from master
    standing_cols = ["team_id", "season_id", "team_points", "team_league_rank",
                     "team_gf", "team_ga", "team_gd", "team_wins", "team_draws",
                     "team_losses", "team_relegated"]
    standing_avail = [c for c in standing_cols if c in df.columns]
    standing_df = df[standing_avail].drop_duplicates(["team_id", "season_id"])
    agg = agg.merge(standing_df, on=["team_id", "season_id"], how="left")

    # --- Observed outcome labels (Stage 17a flags, replacing proxy) ---
    log.info("Joining observed promoted/relegated flags from Stage 17a...")
    agg = join_observed_outcome_flags(agg)

    # --- Departure features ---
    log.info("Computing departure features...")
    dep = compute_departures(df, seasons_sorted)
    dep_rows = []
    for _, r in agg[["team_id", "season_id"]].iterrows():
        key = (r["team_id"], r["season_id"])
        if key in dep:
            dep_rows.append(dep[key])
        else:
            dep_rows.append({"players_departed": np.nan, "departures_value": np.nan,
                             "players_remaining": np.nan, "departure_rate": np.nan})
    dep_df = pd.DataFrame(dep_rows)
    for c in dep_df.columns:
        agg[c] = dep_df[c].values

    # --- Lagged targets ---
    log.info("Computing lagged targets (next-season outcomes)...")
    # Store both the current-season values and a key naming scheme so the
    # same lookup dict can serve current and next values.
    standing_lookup = {}
    for _, r in standing_df.iterrows():
        standing_lookup[(r["team_id"], r["season_id"])] = {
            "team_points_curr": r.get("team_points", np.nan),
            "team_rank_curr": r.get("team_league_rank", np.nan),
            "team_gd_curr": r.get("team_gd", np.nan),
        }

    agg["points_next_season"] = np.nan
    agg["rank_next_season"] = np.nan
    agg["gd_next_season"] = np.nan
    agg["points_delta"] = np.nan
    agg["rank_delta"] = np.nan

    next_season_map = {188124: 189012, 189012: 189974}
    for idx, row in agg.iterrows():
        nxt = next_season_map.get(row["season_id"])
        if nxt is None:
            continue
        key_next = (row["team_id"], nxt)
        key_curr = (row["team_id"], row["season_id"])
        if key_next in standing_lookup and key_curr in standing_lookup:
            nxt_info = standing_lookup[key_next]
            curr_info = standing_lookup[key_curr]
            agg.at[idx, "points_next_season"] = nxt_info["team_points_curr"]
            agg.at[idx, "rank_next_season"] = nxt_info["team_rank_curr"]
            agg.at[idx, "gd_next_season"] = nxt_info["team_gd_curr"]
            if pd.notna(curr_info["team_points_curr"]) and pd.notna(nxt_info["team_points_curr"]):
                agg.at[idx, "points_delta"] = nxt_info["team_points_curr"] - curr_info["team_points_curr"]
            if pd.notna(curr_info["team_rank_curr"]) and pd.notna(nxt_info["team_rank_curr"]):
                agg.at[idx, "rank_delta"] = nxt_info["team_rank_curr"] - curr_info["team_rank_curr"]

    # --- Sort and output ---
    agg = agg.sort_values(["team_id", "season_id"]).reset_index(drop=True)

    log.info("=" * 60)
    log.info("CLUB-SEASON DATASET")
    log.info("=" * 60)
    log.info("Shape: %s", agg.shape)
    log.info("Teams: %d", agg["team_id"].nunique())
    log.info("Seasons: %s", sorted(agg["season_id"].unique()))
    log.info("Club-season pairs: %d", len(agg))
    log.info("Transitions (valid points_next_season): %d", int(agg["points_next_season"].notna().sum()))
    log.info("Columns (%d):", len(agg.columns))
    for c in agg.columns:
        nn = agg[c].notna().sum()
        log.info("  %-30s %d/%d non-null", c, nn, len(agg))

    log.info("\n--- Sample: Real Madrid ---")
    rm = agg[agg["team_name"] == "Real Madrid"]
    if len(rm) > 0:
        log.info(rm[["team_name", "season_id", "team_points", "team_league_rank",
                      "team_relegated_proxy", "team_relegated", "team_promoted",
                      "total_market_value", "squad_size", "count_GK", "count_DF",
                      "count_MF", "count_FW", "total_goals", "total_assists",
                      "goal_concentration_hhi", "players_departed", "departures_value",
                      "points_next_season", "rank_next_season", "points_delta", "rank_delta"
                      ]].to_string(index=False))

    log.info("\n--- Null summary for key fields ---")
    key_fields = ["total_market_value", "avg_age", "total_goals",
                  "goal_concentration_hhi", "team_points", "team_league_rank",
                  "team_relegated_proxy", "team_relegated", "team_promoted",
                  "players_departed", "departures_value",
                  "points_next_season", "rank_next_season", "points_delta"]
    for f in key_fields:
        if f in agg.columns:
            nn = agg[f].notna().sum()
            log.info("  %-30s %d/%d non-null", f, nn, len(agg))

    agg.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved: %s (%d rows x %d cols)", OUTPUT_FILE, *agg.shape)


if __name__ == "__main__":
    main()
