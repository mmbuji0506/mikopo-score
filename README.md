# MikopoScore - an end-to-end credit-risk scoring system

"Mikopo" = Swahili for *loans*. This project builds, validates, serves and monitors a
credit-scoring model the way a bank's data team does, using **synthetic** Tanzanian-style loan
data (real bank data is confidential).

| Bank function | What this project contains |
|---|---|
| Data Architecture | Star-schema warehouse (`sql/schema.sql`, `warehouse.py`), ETL, SQL data-quality rules, load audit log |
| Data Science / Modelling | WoE + logistic-regression **scorecard** (champion) vs gradient boosting (challenger) |
| Model Risk | Independent validation report with AUC / Gini / KS / calibration / PSI / fairness checks |
| MLOps / AI Platform | MLflow tracking, FastAPI service, Docker, GitHub Actions CI, unit tests, drift monitoring |

## Quick start

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1      Windows (cmd): .venv\Scripts\activate.bat
# macOS / Linux:         source .venv/bin/activate
pip install -e ".[dev]"

python -m pytest -q                 # 19 tests pass (21 once the dashboard extra below is installed)
python -m mikopo.pipeline           # generate data -> warehouse -> train -> validate -> monitor
uvicorn mikopo.api:app --reload     # then open http://127.0.0.1:8000/docs
```

Optional visual dashboard (score an applicant, performance, drift alerts, scorecard points, warehouse):

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py      # opens http://localhost:8501
```

Then read `reports/validation_report.md`, and follow **`docs/STEP_BY_STEP.md`** to understand and
defend every file.

Run one step at a time:
`python -m mikopo.data_gen`, `warehouse`, `train`, `validate`, `monitor`.

View experiment runs: `mlflow ui --backend-store-uri mlruns`

## Run with Docker

Install Docker Desktop first (on Windows it needs WSL 2) and wait until it shows "running". Then, in the folder containing `Dockerfile`:

```bash
docker build -t mikopo-score .                  # first build takes a few minutes: it trains the model inside the image
docker run --rm -p 8000:8000 mikopo-score       # API       -> http://localhost:8000/docs
docker run --rm -p 8501:8501 mikopo-score streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
                                                # dashboard -> http://localhost:8501
```

Copy the generated reports out of the image:

```bash
docker create --name tmp mikopo-score
docker cp tmp:/app/reports ./reports_from_docker
docker rm tmp
```

Stop a container with Ctrl+C, or `docker ps` then `docker stop <container-id>`.

## Example API call

```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{
  "age": 29, "employment_type": "informal_trader", "monthly_income_tzs": 450000,
  "loan_amount_tzs": 2500000, "loan_term_months": 12, "existing_loans": 2,
  "months_with_bank": 8, "mobile_money_monthly_txn": 35,
  "mobile_money_avg_balance_tzs": 60000, "past_delinquencies": 1 }'
```

Returns the probability of default, a credit score, a decision (approve / refer / decline) and the
top 3 reasons that cost the applicant points.

## Honest limitations
* The data is simulated: metrics describe the simulator, not real borrowers.
* Validation thresholds are this project's own teaching policy, not any bank's.
* Docker and CI files are included but were written without running Docker or GitHub Actions in the authoring environment.
  Everything else (pipeline, API, tests) was executed end to end.
