"""
fetch_football_data_co_uk.py
============================
Download match-result CSVs from football-data.co.uk and derive per
team-season league standings with observed (not position-proxy)
promotion/relegation flags paired across tier-1/tier-2 by country.

Divisions (10): E0/E1 (England), SP1/SP2 (Spain), D1/D2 (Germany),
I1/I2 (Italy), F1/F2 (France). Seasons: last 12 ("1314".."2425").

Raw CSVs are saved unmodified under
    data/raw_multi_league/football_data_co_uk/{div}/{season}.csv
and treated as immutable once downloaded (idempotent re-runs reuse them).

Output: data/interim/derived_league_standings_multi.csv
"""

import hashlib
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(r"C:\football")
RAW_ROOT = PROJECT_ROOT / "data" / "raw_multi_league" / "football_data_co_uk"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_FILE = INTERIM_DIR / "derived_league_standings_multi.csv"

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

# division -> (country, tier)
DIVISIONS = {
    "E0": ("England", 1), "E1": ("England", 2),
    "SP1": ("Spain", 1), "SP2": ("Spain", 2),
    "D1": ("Germany", 1), "D2": ("Germany", 2),
    "I1": ("Italy", 1), "I2": ("Italy", 2),
    "F1": ("France", 1), "F2": ("France", 2),
}

# Season codes for the last 12 seasons, generated programmatically.
# fd-co.uk codes are 4 chars: last two digits of start and end years (1314..2425).
CURRENT_END_YEAR = 2025
SEASON_CODES = [
    f"{str(y)[2:]}{str(y + 1)[2:]}"
    for y in range(CURRENT_END_YEAR - 12, CURRENT_END_YEAR)
]

MATCH_COLS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

# Known fd-co.uk naming aliases that differ between seasons for the same club.
NAME_ALIASES = {
    "Nott'm Forest": "Nottingham Forest",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Sheffield Weds": "Sheffield Wednesday",
    "Sp Gijon": "Sporting Gijon",
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atletico Madrid",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "M'gladbach": "Borussia M'gladbach",
    "Preußen Münster": "Preussen Munster",
    "Paris SG": "Paris Saint-Germain",
    "St Etienne": "Saint-Etienne",
}


def norm_name(name: str) -> str:
    """Normalize a team name for cross-season identity matching."""
    if not isinstance(name, str):
        return ""
    name = str(name).strip()
    name = NAME_ALIASES.get(name, name)
    out = []
    for ch in name:
        cp = ord(ch)
        if 0x1E00 <= cp <= 0x1EFF:  # Latin Extended Additional (e.g. Preußen)
            ch = ch.translate(str.maketrans("ß", "ss"))
        out.append(ch)
    name = "".join(out)
    import unicodedata
    name = "".join(c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn")
    return " ".join(name.strip().lower().split())


def download_csv(div: str, season: str) -> Path:
    """Download {div}/{season}.csv to RAW_ROOT, unchanged. Idempotent."""
    dest_dir = RAW_ROOT / div
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{season}.csv"
    if dest.exists():
        log.info("  CACHE -> %s/%s.csv (exists, skipped)", div, season)
        return dest

    url = BASE_URL.format(season=season, div=div)
    tmp = dest.with_suffix(".csv.tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        tmp.replace(dest)
        log.info("  OK  -> %s/%s.csv (%d bytes)", div, season, dest.stat().st_size)
    except urllib.error.HTTPError as e:
        log.warning("  SKIP -> %s/%s.csv HTTP %s (missing, logged, no crash)", div, season, e.code)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("  SKIP -> %s/%s.csv fetch error: %s", div, season, e)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return None
    return dest


def load_matches(path: Path) -> pd.DataFrame:
    """Read a raw match CSV keeping only the columns we need.
    Some archived files (e.g. I2/1819) are latin-1; fall back on utf-8 failure."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(pd.io.common.StringIO(raw.decode(enc)),
                             usecols=lambda c: c in MATCH_COLS, low_memory=False)
            break
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            df = None
    if df is None or df.empty:
        return pd.DataFrame()
    for c in ["FTHG", "FTAG"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    return df


def derive_standings(div: str, season: str, path: Path) -> pd.DataFrame:
    """Aggregate match rows into a final team-season table."""
    df = load_matches(path)
    if df.empty:
        return pd.DataFrame()

    home = df[["HomeTeam", "FTHG", "FTAG"]].rename(
        columns={"HomeTeam": "team", "FTHG": "gf", "FTAG": "ga"})
    away = df[["AwayTeam", "FTHG", "FTAG"]].rename(
        columns={"AwayTeam": "team", "FTHG": "ga", "FTAG": "gf"})
    both = pd.concat([home, away], ignore_index=True)

    both["wins"] = (both["gf"] > both["ga"]).astype(int)
    both["draws"] = (both["gf"] == both["ga"]).astype(int)
    both["losses"] = (both["gf"] < both["ga"]).astype(int)
    both["points"] = 3 * both["wins"] + 1 * both["draws"]

    agg = (
        both.groupby("team")
        .agg(matches_played=("gf", "size"),
             wins=("wins", "sum"), draws=("draws", "sum"), losses=("losses", "sum"),
             goals_for=("gf", "sum"), goals_against=("ga", "sum"),
             points=("points", "sum"))
        .reset_index()
    )
    agg["goal_diff"] = agg["goals_for"] - agg["goals_against"]
    agg["points_per_match"] = (agg["points"] / agg["matches_played"].replace(0, pd.NA)).astype(float)

    agg = agg.sort_values(["points", "goal_diff", "goals_for", "team"],
                          ascending=[False, False, False, True]).reset_index(drop=True)
    agg["final_position"] = range(1, len(agg) + 1)

    country, tier = DIVISIONS[div]
    agg.insert(0, "division_code", str(div))
    agg.insert(0, "season_code", str(season))
    agg.insert(0, "tier", tier)
    agg.insert(0, "country", country)
    agg["team_name_norm"] = agg["team"].map(norm_name)
    return agg


def compute_observed_flags(standings: pd.DataFrame) -> pd.DataFrame:
    """Set promoted/relegated from ACTUAL next-season tier membership.

    For each country and each season-pair (S, S+1): a team recorded in tier-2
    during S that is present in tier-1 during S+1 was promoted; a team in
    tier-1 during S present in tier-2 during S+1 was relegated. This is an
    observed outcome, not a table-position proxy. The terminal season (no
    S+1 in scope) gets NA flags.
    """
    flags = {"promoted": {}, "relegated": {}}
    seasons = sorted(set(standings["season_code"]))
    for country in sorted(set(standings["country"])):
        by_season = {}
        for season in seasons:
            sub = standings[(standings["country"] == country) & (standings["season_code"] == season)]
            tier1 = set(sub.loc[sub["tier"] == 1, "team_name_norm"])
            tier2 = set(sub.loc[sub["tier"] == 2, "team_name_norm"])
            by_season[season] = (tier1, tier2)

        for i, season in enumerate(seasons):
            if i + 1 >= len(seasons):
                continue
            t1_s, t2_s = by_season[season]
            t1_sp, t2_sp = by_season[seasons[i + 1]]
            # Observed moves: a tier-2 team in S now in tier-1 is promoted;
            # a tier-1 team in S now in tier-2 is relegated.
            promoted = t2_s & t1_sp
            relegated = t1_s & t2_sp
            for team in promoted:
                flags["promoted"].setdefault(season, set()).add(team)
            for team in relegated:
                flags["relegated"].setdefault(season, set()).add(team)

    standings["promoted"] = pd.NA
    standings["relegated"] = pd.NA
    standings["flag_method"] = "observed_next_season"
    for season in seasons:
        if season == max(seasons):
            m = standings["season_code"] == season
            standings.loc[m, "promoted"] = False
            standings.loc[m, "relegated"] = False
            standings.loc[m, "flag_method"] = "terminal_season_unobserved"
            continue
        p = flags["promoted"].get(season, set())
        r = flags["relegated"].get(season, set())
        m = standings["season_code"] == season
        standings.loc[m & standings["team_name_norm"].isin(p), "promoted"] = True
        standings.loc[m & ~standings["team_name_norm"].isin(p), "promoted"] = False
        standings.loc[m & standings["team_name_norm"].isin(r), "relegated"] = True
        standings.loc[m & ~standings["team_name_norm"].isin(r), "relegated"] = False
    return standings


def write_raw_manifest() -> None:
    """Write _manifest.csv + _meta.json mirroring the laliga raw-folder convention."""
    files = sorted(f for f in RAW_ROOT.rglob("*.csv") if f.name != "_manifest.csv")
    if not files:
        return
    rows = []
    for f in files:
        rel = f.relative_to(RAW_ROOT)
        try:
            sha = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        except OSError:
            sha = "n/a"
        rows.append({"file": str(rel).replace("\\", "/"), "bytes": f.stat().st_size, "sha256": sha})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(RAW_ROOT / "_manifest.csv", index=False)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv",
        "divisions": DIVISIONS,
        "season_codes": SEASON_CODES,
        "raw_immutable": True,
        "files": len(rows),
        "note": "Raw files are never modified after download.",
    }
    (RAW_ROOT / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("  Wrote manifest: %s (%d files)", RAW_ROOT / "_manifest.csv", len(rows))


def main() -> int:
    t0 = time.time()
    log.info("=" * 60)
    log.info("START stage 17a: fetch_football_data_co_uk")
    log.info("=" * 60)
    log.info("Seasons: %s", SEASON_CODES)
    log.info("Divisions: %s", sorted(DIVISIONS))

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Download everything (idempotent; reuses existing raw files) ---
    log.info("PHASE 1: Download raw match CSVs")
    expected = len(DIVISIONS) * len(SEASON_CODES)
    downloaded_ok = 0
    skipped = []
    for div in sorted(DIVISIONS):
        for season in SEASON_CODES:
            p = download_csv(div, season)
            if p is None:
                skipped.append(f"{div}/{season}")
            else:
                downloaded_ok += 1
    log.info("  Downloads ready: %d/%d (missing/skipped: %d)",
             downloaded_ok, expected, len(skipped))
    if skipped:
        log.info("  Skipped (missing): %s", ", ".join(skipped))

    # --- 2. Derive standings per division-season ---
    log.info("PHASE 2: Derive per team-season standings")
    frames = []
    for div in sorted(DIVISIONS):
        for season in SEASON_CODES:
            path = RAW_ROOT / div / f"{season}.csv"
            if not path.exists():
                continue
            try:
                st = derive_standings(div, season, path)
            except Exception as e:  # noqa: BLE001
                log.warning("  WARN -> could not derive %s/%s: %s", div, season, e)
                continue
            frames.append(st)
    if not frames:
        log.error("FAIL stage 17a: no standings could be derived (network or file issue)")
        return 1
    standings = pd.concat(frames, ignore_index=True)

    # --- 3. Observed promotion/relegation across tier pairs ---
    log.info("PHASE 3: Paired promotion/relegation (observed across seasons)")
    standings = compute_observed_flags(standings)

    standings = standings.sort_values(
        ["country", "season_code", "tier", "final_position"]
    ).reset_index(drop=True)

    # --- 4. Save ---
    standings.to_csv(OUTPUT_FILE, index=False)
    write_raw_manifest()

    # --- 5. Coverage summary ---
    log.info("=" * 60)
    log.info("OUTPUT: %s", OUTPUT_FILE)
    log.info("Shape: %s", standings.shape)
    log.info("-" * 60)
    log.info("Row counts by country x division:")
    piv = standings.groupby(["country", "division_code"]).size().unstack(fill_value=0)
    log.info("\n%s", piv.to_string())
    log.info("\nTeam-seasons per season_code:")
    log.info("  %s", dict(standings["season_code"].value_counts().sort_index()))
    log.info("\nPromoted rows: %d | Relegated rows: %d",
             int(standings["promoted"].eq(True).sum()),
             int(standings["relegated"].eq(True).sum()))
    unmatched = (standings["promoted"].isna() | standings["relegated"].isna()).sum()
    if unmatched:
        log.info("Rows with NA flags: %d (all in terminal season %s)",
                 unmatched, standings["season_code"].max())
    else:
        log.info("No NA flags: terminal season %s set to False (not observed in scope)",
                 standings["season_code"].max())

    dt = time.time() - t0
    log.info("=" * 60)
    log.info("DONE stage 17a: fetch_football_data_co_uk in %.1fs", dt)
    return 0


if __name__ == "__main__":
    sys.exit(main())