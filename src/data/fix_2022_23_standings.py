"""
fix_2022_23_standings.py
========================
Add hardcoded 2022/23 LaLiga final standings to derived_league_standings.csv.
Source: Wikipedia "2022-23 La Liga" (official final table).
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"C:\football")
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
STANDINGS_FILE = INTERIM_DIR / "derived_league_standings.csv"

SEASON_ID_2223 = 188124
SUPER_SEASON_ID_2223 = 2023
SEASON_NAME_2223 = "2022/2023"

# Official 2022/23 LaLiga final standings (Wikipedia verified)
# Columns: team_id, team_name, W, D, L, GF, GA, GD, Pts, rank, relegated
STANDINGS_2223 = [
    (676, "Barcelona",       28, 4,  6,  70, 20,  50, 88, 1,  0),
    (675, "Real Madrid",     24, 6,  8,  75, 36,  39, 78, 2,  0),
    (679, "Atlético Madrid", 23, 8,  7,  70, 33,  37, 77, 3,  0),
    (687, "Real Sociedad",   21, 8,  9,  51, 35,  16, 71, 4,  0),
    (682, "Villarreal",      19, 7,  12, 59, 40,  19, 64, 5,  0),
    (684, "Real Betis",      17, 9,  12, 46, 41,   5, 60, 6,  0),
    (681, "Osasuna",         15, 8,  15, 37, 42,  -5, 53, 7,  0),
    (678, "Athletic Club",   14, 9,  15, 47, 43,   4, 51, 8,  0),
    (686, "Mallorca",        14, 8,  16, 37, 43,  -6, 50, 9,  0),
    (756, "Girona",          13, 10, 15, 58, 55,   3, 49, 10, 0),
    (713, "Rayo Vallecano",  13, 10, 15, 45, 53,  -8, 49, 11, 0),
    (680, "Sevilla",         13, 10, 15, 47, 54,  -7, 49, 12, 0),
    (692, "Celta de Vigo",   11, 10, 17, 43, 53, -10, 43, 13, 0),
    (700, "Cádiz",           10, 12, 16, 30, 53, -23, 42, 14, 0),
    (698, "Getafe",          10, 12, 16, 34, 45, -11, 42, 15, 0),
    (674, "Valencia",        11, 9,  18, 42, 45,  -3, 42, 16, 0),
    (708, "Almería",         11, 8,  19, 49, 65, -16, 41, 17, 0),
    (690, "Real Valladolid", 11, 7,  20, 33, 63, -30, 40, 18, 1),
    (691, "Espanyol",         8, 13, 17, 52, 69, -17, 37, 19, 1),
    (702, "Elche",            5, 10, 23, 30, 67, -37, 25, 20, 1),
]


def main():
    df = pd.read_csv(STANDINGS_FILE)

    new_rows = []
    for (tid, tname, w, d, l, gf, ga, gd, pts, rank, rel) in STANDINGS_2223:
        mp = w + d + l
        ppm = pts / mp if mp > 0 else 0.0
        new_rows.append({
            "season_id": SEASON_ID_2223,
            "super_season_id": SUPER_SEASON_ID_2223,
            "name": SEASON_NAME_2223,
            "team_id": tid,
            "team_name": tname,
            "matches_played": mp,
            "wins": w,
            "draws": d,
            "losses": l,
            "goals_for": gf,
            "goals_against": ga,
            "goal_diff": gd,
            "points": pts,
            "points_per_match": round(ppm, 6),
            "league_rank": rank,
            "relegated": rel,
        })

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([new_df, df], ignore_index=True)
    combined = combined.sort_values(["season_id", "league_rank"]).reset_index(drop=True)

    combined.to_csv(STANDINGS_FILE, index=False)
    print(f"Added {len(new_rows)} rows for 2022/23. Total: {len(combined)} rows")
    print(f"Seasons: {sorted(combined['season_id'].unique())}")
    print(f"Teams per season:")
    for sid in sorted(combined["season_id"].unique()):
        n = len(combined[combined["season_id"] == sid])
        print(f"  {sid}: {n} teams")


if __name__ == "__main__":
    main()
