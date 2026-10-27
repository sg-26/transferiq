# TransferIQ — Football Squad Optimization & Decision Support

TransferIQ is a football analytics research project that explores the trade-off between **raising transfer funds and maintaining sporting performance**. It combines player-level modelling, squad constraints, transfer-fee estimates, and scenario analysis to compare potential player-sale decisions.

The project focuses on Real Madrid's 2024/25 squad as its primary scenario-planning example, alongside broader LaLiga player and club-season analytics.

## Key Features

- **Sporting risk modelling:** Estimates the risk of a player's goal-contribution decline in a subsequent season.
- **Club performance modelling:** Uses squad-level features to estimate goals scored, goals conceded, and expected points.
- **Transfer scenario optimization:** Generates and compares squad-sale scenarios while enforcing minimum positional depth.
- **Transfer pricing:** Estimates potential sale proceeds using market values and a limited set of observed Real Madrid transfer fees.
- **Net financial comparison:** Considers sale proceeds and estimated avoided wage costs when comparing scenarios.
- **Interactive dashboard:** Provides scenario planning, comparisons, La Fábrica analysis, financial views, and methodology disclosures.
- **Data validation:** Includes checks and documentation for transfer data coverage, identity resolution, and modelling limitations.

## Example Results

The current Real Madrid 2024/25 scenario analysis produced examples such as:

| Scenario | Estimated sale proceeds | Estimated expected-points change |
|---|---:|---:|
| Vinícius Júnior, Jude Bellingham and Federico Valverde sold | €484.5M | −18.8 |
| Jude Bellingham, Kylian Mbappé and Federico Valverde sold | €456.0M | −24.5 |

These are model-generated estimates, not observed outcomes or predictions guaranteed to occur. Financial estimates depend on the assumptions and limited transfer-fee data available.

## Technology

- Python
- Pandas and NumPy
- LightGBM and scikit-learn
- Streamlit and Plotly
- Parquet-based data processing

## Repository Structure

```text
transferiq/
├── docs/
│   ├── DATA_AUDIT.md
│   ├── TRANSFER_DATA_ACQUISITION.md
│   ├── TRANSFER_DATA_COVERAGE.md
│   ├── feature_importance.csv
│   └── sale_sanity_check.json
├── src/
│   ├── __init__.py
│   ├── dashboard/
│   │   └── app.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── build_club_season_dataset.py
│   │   ├── build_master_dataset.py
│   │   ├── extract_standings.py
│   │   ├── fetch_football_data_co_uk.py
│   │   ├── fix_2022_23_standings.py
│   │   ├── integrate_financials_and_la_fabrica.py
│   │   └── validate_transfer_data.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── club_sporting_model.py
│   │   └── sporting_risk_model.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── pareto_engine.py
│   │   └── scenario_pareto.py
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── extend_master_pricing.py
│   │   ├── price_scenarios.py
│   │   ├── sale_price_model.py
│   │   └── sale_sanity_check.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── scenario_scoring.py
│   └── simulation/
│       ├── __init__.py
│       ├── buy_side.py
│       ├── simulate_scenarios.py
│       └── squad_counterfactual.py
├── .gitignore
├── README.md
├── requirements.txt
└── run_pipeline.py
```

## Data and Reproducibility

The repository contains the Python source code and selected documentation and evaluation outputs. The full raw datasets, processed data, and trained model files are not included.

Consequently, the complete pipeline may require additional data files and model-training steps before it can run. The data acquisition and coverage documents describe the current data sources, scope, and outstanding requirements.

## Limitations

- The transfer-event dataset currently covers Real Madrid rather than all 24 LaLiga clubs.
- The sale-fee model is based on a small sample of observed Real Madrid permanent out-sales.
- Wage and contract-length estimates are heuristic rather than verified player-level financial records.
- Identity matching between transfer records and the internal player dataset remains incomplete.
- Sporting models use a limited historical window, and scenario-level performance changes are estimates.
- Financial and sporting results should be interpreted as research outputs, not professional transfer advice.

See the files in [`docs/`](docs/) for the data acquisition contract, coverage report, audit findings, and selected model evaluation outputs.

## Project Status

TransferIQ is an ongoing football analytics research project. Future work includes expanding transfer coverage across LaLiga, improving player identity resolution, collecting reliable wage and contract data, and strengthening model validation.
