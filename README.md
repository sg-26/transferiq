# TransferIQ
### Football Squad Optimization Through Sporting Risk and Profitability Analysis

TransferIQ is a football analytics and decision-support system that evaluates player-sale strategies by balancing **financial returns against potential sporting losses**.

The project approaches transfers as a squad-level optimization problem rather than simply ranking individual players by market value. It combines predictive modelling, expected-points simulation, transfer-fee estimation, scenario evaluation, and Pareto-frontier analysis to explore how clubs can raise funds while preserving sporting performance.

## The Problem

When a football club needs to generate revenue through player sales, the players with the highest market values are not necessarily the best candidates to sell. Their departure may also affect squad depth, attacking output, and expected league points.

TransferIQ explores the question:

**How can a club raise a target amount through player sales while minimizing the predicted loss in sporting performance?**

## Key Features

- **Sporting Risk Modelling:** Estimates the risk of a player's goal contribution declining in the following season.
- **Club Performance Modelling:** Models expected points using attacking and defensive performance estimates.
- **Squad Counterfactuals:** Simulates the effect of selling different combinations of players while enforcing minimum squad-composition constraints.
- **Transfer-Fee Estimation:** Estimates potential sale fees from available market-value and historical transfer information.
- **Pareto Optimization:** Identifies non-dominated sale scenarios that balance financial returns against predicted sporting losses.
- **Reinvestment Analysis:** Examines whether reinvesting sale proceeds could recover some of the predicted sporting impact.
- **Academy Financial Analysis:** Studies selected La Fábrica graduates and their historical transfer-fee contribution.
- **Interactive Dashboard:** Provides a Streamlit interface for planning sales, building scenarios, comparing outcomes, exploring academy-related financial analysis, and reviewing modelling assumptions.
- **Multi-League Data Preparation:** Includes a data-collection component for football results across multiple leagues and seasons.

## How It Works

The system follows a multi-stage pipeline:

1. **Data preparation:** Processes football match, standings, player, transfer, and financial data.
2. **Feature engineering:** Builds player-level and club-level analytical features.
3. **Predictive modelling:** Estimates individual performance-decline risk and club sporting performance.
4. **Scenario generation:** Evaluates possible combinations of players to sell under squad constraints.
5. **Financial estimation:** Calculates estimated transfer proceeds and, where supported, net financial impact.
6. **Sporting simulation:** Estimates changes in expected league points following squad changes.
7. **Pareto optimization:** Identifies scenarios offering different trade-offs between revenue and sporting risk.
8. **Dashboard presentation:** Makes the results accessible through an interactive Streamlit application.

## Selected Results

The current implementation includes the following evaluation results:

- **Individual performance-decline model:** LightGBM achieved an ROC-AUC of approximately **0.824** on the project's temporal evaluation split.
- **Club expected-points model:** Leave-one-out evaluation across 34 club-season transitions produced an \(R^2\) of approximately **0.834**, an RMSE of **7.33 points**, and an MAE of **5.96 points**.
- **Scenario evaluation:** The pipeline evaluated thousands of squad-sale scenarios and used Pareto analysis to identify revenue–sporting-impact trade-offs.
- **Academy analysis:** A curated set of 20 La Fábrica graduates was examined alongside 137 historical cases spanning 2006–2026.

These are project-specific evaluation results, not guarantees of real-world predictive performance. Their interpretation depends on the data, validation design, and assumptions described below.

## Technology Stack

- **Language:** Python 3.11
- **Data processing:** Pandas, NumPy, PyArrow
- **Machine learning:** LightGBM, scikit-learn
- **Modelling and analysis:** Ridge regression, classification, expected-points simulation
- **Optimization:** Pareto-frontier analysis and scenario scoring
- **Visualization:** Plotly, Matplotlib
- **Dashboard:** Streamlit
- **Data formats:** CSV, Parquet, Excel, SQL

## Project Structure

```text
football/
├── src/
│   ├── data/           # Data extraction, preparation and validation
│   ├── features/       # Feature engineering
│   ├── models/         # Sporting-risk and club-performance models
│   ├── simulation/     # Squad-sale and reinvestment simulations
│   ├── pricing/        # Transfer-fee estimation and pricing checks
│   ├── optimization/   # Pareto optimization and scenario analysis
│   ├── reporting/      # Scenario scoring and reporting
│   └── dashboard/     # Streamlit dashboard
├── data/               # Raw, interim and processed datasets
├── models/             # Trained models and evaluation artefacts
├── outputs/             # Generated figures and other outputs
├── run_pipeline.py      # Main pipeline entry point
├── README.md
└── .gitignore
```

The local development environment (`.venv/`), Python cache directories, and large raw-data files are not intended to be committed to the repository.

## Running the Project

The project was developed using Python 3.11. Install the dependencies required by the pipeline and dashboard in a separate virtual environment.

Once the required data files and dependencies are available, the main pipeline can be run with:

```bash
python run_pipeline.py
```

To run the dashboard, use the relevant dashboard entry point:

```bash
streamlit run src/dashboard/app.py
```

The dashboard also includes a self-check mode:

```bash
python src/dashboard/app.py --selfcheck
```

**Note:** Full reproduction requires the appropriate input datasets and generated artefacts. Large raw datasets are not distributed with this repository, and the complete pipeline may require additional data acquisition and configuration.

## Limitations and Assumptions

- **Limited transfer-fee training data:** The current fee model was trained on a small sample of non-free Real Madrid out-transfers. It should not be treated as a universal transfer-valuation model.
- **Club-specific financial coverage:** Financial assumptions and fee estimates are not equally well supported for every club.
- **Simplified sporting-impact estimation:** Scenario deltas are primarily attack-driven because the current defensive-performance mapping is less reliable.
- **Market value is not transfer fee:** Estimated fees depend on historical data and modelling assumptions; actual fees may differ substantially.
- **Scenario-dependent results:** Outputs depend on the selected squad constraints, available player data, and simulation assumptions.
- **Academy linkage:** The La Fábrica analysis uses a curated set of graduates and historical cases, so it does not represent a complete automated linkage of every academy player.
- **Data availability:** Reproducing results depends on access to the required datasets and the availability of compatible data sources.

TransferIQ is intended as an **analytical decision-support tool**, not as a substitute for club scouting, financial due diligence, or professional sporting judgement.

## Future Work

Potential improvements include stronger defensive-impact modelling, broader transfer-fee training data, improved player-identity matching across datasets, multi-season optimization, position-specific squad constraints, and more detailed wage and reinvestment modelling.

## Project Status

Research and portfolio project. The pipeline, models, scenario analysis, and dashboard form an evolving system for exploring the relationship between football-finance decisions and sporting performance.
