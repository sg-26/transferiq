"""
run_pipeline.py -- End-to-end reproduction of the SSAC player-sales research engine.

Executes every pipeline stage sequentially from C:\\football, logs per-stage and
total timing, and aborts with a clear error if any stage fails. The pipeline
regenerates all interim and processed artifacts consumed by the Streamlit
decision dashboard (src/dashboard/app.py).

Console compatibility: Windows cp1252. Log messages use '->' instead of Unicode
arrows, and no box-drawing/emoji characters are printed.

Stages (order matters -- each consumes the previous stage's outputs):
  1  src/data/extract_standings.py              standings from SQL dump
  2  src/data/fix_2022_23_standings.py          official 2022/23 table (complete)
  3  src/data/build_master_dataset.py           1805 x 370 player-season master
  4  src/features/build_features.py             1805 x 402 features
  5  src/models/sporting_risk_model.py          player-level decline (LightGBM)
  6  src/simulation/simulate_scenarios.py       market-value / GC Monte Carlo
  7  src/data/build_club_season_dataset.py      60 club-seasons (34 transitions)
  8  src/models/club_sporting_model.py          structural xP club model + LOOCV
  9  src/simulation/squad_counterfactual.py     RM 24/25 sale scenario enumeration
  10 src/optimization/scenario_pareto.py        model-driven scenario Pareto
  11 src/data/integrate_financials_and_la_fabrica.py  financial + La Fabrica
  12 src/pricing/sale_price_model.py          learned sale-price + wage model
  13 src/pricing/extend_master_pricing.py     master + pricing/wage/contract cols
  14 src/pricing/price_scenarios.py           predicted fee + net impact per scenario
  15 src/simulation/buy_side.py               sell+buy net scenarios + net frontier
  16 src/pricing/sale_sanity_check.py          honest sanity check of the sale-price engine
  17a src/data/fetch_football_data_co_uk.py     multi-league standings (football-data.co.uk)

Usage:
    python run_pipeline.py [--skip <stage_number,stage_number,...>]
                           [--from <stage_number>]
                           [--dashboard]
Example:
    python run_pipeline.py --skip 5,6        # reuse existing model/simulation outputs
    python run_pipeline.py --from 9          # only scenario & Pareto stages
    python run_pipeline.py --dashboard       # launch the Streamlit dashboard instead
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\football")
PY = Path(r"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe")

DASHBOARD = "src/dashboard/app.py"

STAGES = [
    ("extract_standings", "src/data/extract_standings.py"),
    ("fix_2022_23_standings", "src/data/fix_2022_23_standings.py"),
    ("build_master_dataset", "src/data/build_master_dataset.py"),
    ("build_features", "src/features/build_features.py"),
    ("sporting_risk_model", "src/models/sporting_risk_model.py"),
    ("simulate_scenarios", "src/simulation/simulate_scenarios.py"),
    ("build_club_season_dataset", "src/data/build_club_season_dataset.py"),
    ("club_sporting_model", "src/models/club_sporting_model.py"),
    ("squad_counterfactual", "src/simulation/squad_counterfactual.py"),
    ("scenario_pareto", "src/optimization/scenario_pareto.py"),
    ("integrate_financials_la_fabrica", "src/data/integrate_financials_and_la_fabrica.py"),
    ("sale_price_model", "src/pricing/sale_price_model.py"),
    ("extend_master_pricing", "src/pricing/extend_master_pricing.py"),
    ("price_scenarios", "src/pricing/price_scenarios.py"),
    ("buy_side", "src/simulation/buy_side.py"),
    ("sale_sanity_check", "src/pricing/sale_sanity_check.py"),
    ("fetch_football_data_co_uk", "src/data/fetch_football_data_co_uk.py"),
]


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [pipeline] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full SSAC pipeline.")
    ap.add_argument("--skip", default="", help="comma-separated 1-based stage numbers to skip")
    ap.add_argument("--from", dest="start", type=int, default=1,
                    help="start from this 1-based stage number")
    ap.add_argument("--dashboard", action="store_true",
                    help="launch the Streamlit dashboard (src/dashboard/app.py)")
    args = ap.parse_args()

    if args.dashboard:
        dashboard = ROOT / DASHBOARD
        if not dashboard.exists():
            log(f"FAIL: dashboard script not found: {DASHBOARD}")
            return 1
        log(f"Launching dashboard: {DASHBOARD}")
        return subprocess.run(
            [str(PY), "-m", "streamlit", "run", str(dashboard)],
            cwd=str(ROOT),
        ).returncode

    skipped = {int(x) for x in args.skip.split(",") if x.strip().isdigit()}
    t0 = time.time()
    total_ok = 0
    total_failed = 0

    for idx, (name, rel) in enumerate(STAGES, start=1):
        if idx < args.start:
            log(f"SKIP (from={args.start}) stage {idx}/{len(STAGES)}: {name}")
            continue
        if idx in skipped:
            log(f"SKIP (user)         stage {idx}/{len(STAGES)}: {name}")
            continue

        script = ROOT / rel
        if not script.exists():
            log(f"FAIL stage {idx}: script not found: {rel}")
            total_failed += 1
            continue

        s = time.time()
        log(f"START stage {idx}/{len(STAGES)}: {name} ({rel})")
        try:
            proc = subprocess.run(
                [str(PY), str(script)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            log(f"FAIL stage {idx}: {name} timed out after 30 min")
            total_failed += 1
            break

        elapsed = time.time() - s
        if proc.returncode != 0:
            log(f"FAIL stage {idx}: {name} rc={proc.returncode} in {elapsed:.1f}s")
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
            for line in tail:
                log(f"    | {line}")
            total_failed += 1
            break  # stop the chain on the first hard failure

        total_ok += 1
        log(f"DONE stage {idx}: {name} in {elapsed:.1f}s (rc=0)")

        # surface any non-fatal warnings carried on stderr
        warn = (proc.stderr or "").strip()
        if warn:
            for line in warn.splitlines()[-4:]:
                log(f"  note: {line}")

    dt = time.time() - t0
    log("-" * 56)
    log(f"PIPELINE COMPLETE: {total_ok} ok, {total_failed} failed, total {dt:.1f}s")
    log("Run the dashboard with: streamlit run src/dashboard/app.py")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())