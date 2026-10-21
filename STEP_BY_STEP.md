# MikopoScore: Step-by-Step Guide (build it, understand it, defend it)

This guide walks through the project in the order you should **build and learn** it. Every code file
also has comments on every non-obvious line. This guide explains the *logic and reasons* behind them, so
you can answer "why did you do that?" in front of a supervisor or a bank interviewer.

All numbers below come from a real run of this exact code (seed 42).

---

## 0. The big picture

**The business problem.** A bank gets a loan application. It must decide: *approve, refer to a human, or decline?*
The key quantity is the **Probability of Default (PD)**: the chance this borrower will fail to repay.

**What you build** (each part maps to a CRDB job family):

| Part | Files | Job family it proves |
|---|---|---|
| Data warehouse + data quality | `sql/schema.sql`, `warehouse.py` | Data Architecture |
| Scorecard model | `scorecard.py`, `train.py` | Data Science / ML |
| Validation report | `validate.py` | Model Risk Analyst |
| API, tests, Docker, CI, MLflow, monitoring | `api.py`, `tests/`, `Dockerfile`, `monitor.py` | MLOps / AI Platform |

**The data flow:**

```
data_gen.py -> applications.csv -> warehouse.py -> warehouse.db (star schema + quality checks)
                    |
                    v
                train.py -> scorecard.joblib + challenger.joblib + model_data.csv
                    |
        +-----------+-----------+
        v                       v
   validate.py             monitor.py             api.py (serves the saved scorecard)
 (validation_report.md)  (drift alerts + chart)
```

**Honesty rule.** The data is *simulated*. Say this first in any presentation. It is a strength when you
state it, and a weakness only if someone discovers it.

---

## 1. Setup and first run

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell   (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"             # installs the project and its libraries
python -m pytest -q                 # expect: 19 passed (21 if you also install the dashboard extra)
python -m mikopo.pipeline           # runs all 5 steps
```

`pip install -e .` ("editable install") makes `import mikopo` work anywhere and lets `config.py`
find the project folder. Always run commands from the project root.

---

## 2. Step-by-step: what every file does and why

### Step 1: `config.py` (settings in one place)
**What:** paths, the random seed, dates, the target column, feature lists, scorecard scaling, decision
cut-offs, validation thresholds.
**Why:** in a bank a model is a *controlled asset*. An auditor must see every number that changes its
behaviour in one place. It also makes experiments reproducible.
**Key lines:**
- `ROOT = Path(__file__).resolve().parents[2]` finds the project folder from the file's own location
  (`mikopo` -> `src` -> project root), so paths work on any computer.
- `RANDOM_SEED = 42` means anyone re-running gets the **identical** dataset and results.
- `OOT_CUTOFF = "2026-01-01"` defines the out-of-time sample (see Step 7).
- `MONITOR_ONLY = ["gender", "region"]` are kept for fairness checks but are **never** model inputs.

### Step 2: `features.py` (one formula, used by training AND the API)
**What:** `monthly_payment()` and `compute_dti()`.
`dti = monthly instalment / monthly income`, where instalment = `loan x (1 + 18% x years) / months`.
**Why:** *training-serving skew* is when production computes a feature differently from training. The model
then silently makes worse decisions and nothing crashes. One shared function makes it impossible.
**Key lines:** `np.where(income > 0, dti, np.nan)` turns zero or negative income into "missing" instead of
creating infinity.

### Step 3: `data_gen.py` (the simulator)
**What:** creates 20,000 loan applications over two years.
**How it works, in order:**
1. Random application dates, plus `t` (0 to 1) used to create drift over time.
2. Demographics: age, gender, region, employment type.
3. Income depends on job type (log-normal, so it has a realistic long right tail), and rises about 4% over the period.
4. Loan amount (1 to 9 months of income) and term (6/12/24/36 months).
5. Credit history, and mobile-money activity (the "alternative data" for informal borrowers). Mobile-money use grows about 50% over time, which creates **drift on purpose**.
6. `dti` via the shared formula.
7. **The hidden truth:** `linear = 1.8*dti + 0.55*past_delinq + 0.20*existing_loans - ... + noise`. This is the secret recipe for who defaults. Positive means riskier.
8. `_calibrate_intercept()` uses bisection (halving a search range 60 times) to pick the intercept so the average default rate is 12%.
9. `default = random < sigmoid(linear + intercept)`. A **sigmoid** converts any number into a probability.
10. Blank out 3% of income and 8% of mobile-money records, because real data is messy.

**If asked "why simulate?"** Real bank data is confidential. A simulator lets you *test the whole pipeline*
and check the model can recover rules you know are true. Real data with the same columns can be dropped in later.

### Step 4: `sql/schema.sql` + `warehouse.py` (data architecture)
**Star schema:** one **fact** table (`fact_loan_application`, the measurements) surrounded by **dimension**
tables (`dim_customer`, `dim_date`, `dim_loan_product`, the context). Analysts answer questions with simple joins.

**ETL order matters:** dimensions are loaded first because the fact table holds *foreign keys* that must
point to existing rows. `PRAGMA foreign_keys = ON` makes SQLite enforce this.

**Surrogate keys:** `customer_key = 1, 2, 3...` are warehouse-owned IDs, independent of the source
system's `customer_id` (the *business key*). This protects the warehouse if the source changes its ID format.

**`dim_date`** has one row for *every* day, even days with no loans, so time-series reports have no gaps.

**Data-quality rules** (`QUALITY_RULES`): each rule is SQL that counts *violating rows*; 0 means pass.
There are 9 rules (7 critical, 2 warnings): duplicates, orphan keys, non-positive amounts, impossible ages,
negative income, implausible `dti`, missingness above 5%, and a row-count reconciliation against the audit log.
If a *critical* rule fails, the pipeline stops. `tests/test_warehouse.py` proves the rules catch a corrupted row.

**`meta_load_audit`** records what was loaded and when. That is *lineage*, which auditors request.

Example business question the star schema answers (`reports/example_query.csv`): *farmers on short-term loans
default at 32.1%, versus 5.1% for salaried borrowers on long terms.*

**Portability:** the SQL is plain enough to run unchanged on PostgreSQL. SQLite is used so it runs on any laptop.

### Step 5: `metrics.py` (how we measure a model)
Memorise these in plain English:

| Metric | Meaning | Good value |
|---|---|---|
| **AUC** | Pick a random defaulter and a random good customer. How often does the model rate the defaulter riskier? | 0.5 = coin flip, 1.0 = perfect |
| **Gini** | `2 x AUC - 1`, a rescaled AUC | higher is better |
| **KS** | Biggest gap between the cumulative share of bads and goods when ranked by risk | higher is better |
| **Brier** | Mean squared error of the probabilities | lower is better |
| **Calibration** | When the model says 12%, do about 12% really default? | predicted ~ observed |
| **PSI** | How much a distribution shifted: `sum((actual% - expected%) x ln(actual%/expected%))` | <0.10 stable, 0.10-0.25 watch, >0.25 act |

**Discrimination vs calibration:** AUC tells you if the model *ranks* people correctly. Calibration tells you if the
*probabilities* are right. A bank needs both, because provisions and pricing use the actual probabilities.

**Key lines:** in `psi_numeric`, `edges` come from quantiles of the **reference** sample, and missing values get
their own bucket, so a jump in missingness is also caught. `np.clip(..., 1e-4, None)` avoids `log(0)`.

### Step 6: `scorecard.py` (the champion model)
**Why a scorecard?** (1) Every decision is a sum of points, so you can tell a customer *why* they were declined,
which regulators require. (2) It is stable and easy to validate. (3) On tabular data it is often close to boosting.

**6a. Binning and WoE (`WoEBinner`).**
For each feature, split values into bins (5 quantile bins for continuous values; one bin per value when a column has 6 or fewer distinct values; `MISSING` as its own bin). Then, per bin:

```
WoE = ln( share of all GOOD customers in the bin / share of all BAD customers in the bin )
```

- WoE > 0 means safer than average, WoE < 0 means riskier.
- Example from this run: `dti > 0.74` has bad rate 25% and WoE -0.87; `dti <= 0.20` has bad rate 6% and WoE +0.71.
- `+0.5` smoothing avoids `ln(0)` when a bin has no bads.
- **Information Value** `IV = sum((good% - bad%) x WoE)` scores a whole feature. Under 0.02 is useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, and above 0.5 is suspicious (check for leakage).

Results: `dti` 0.336, `loan_term_months` 0.211, `past_delinquencies` 0.157, `employment_type` 0.093,
`existing_loans` 0.063, ... `mobile_money_avg_balance_tzs` 0.020 (borderline).

**Why WoE?** It makes each feature linear in the log-odds (what logistic regression assumes), handles outliers and
missing values automatically, and produces a readable points table.

**Binning is learned on TRAIN only.** Fitting on all data would leak information from the future into the model.

**6b. Logistic regression.** It learns `ln(p/(1-p)) = b0 + b1 x WoE1 + b2 x WoE2 + ...`. The coefficients `b` come out
negative because higher WoE (safer) lowers default probability.

**6c. Points scaling.** Banks want scores, not probabilities. Convention: *600 points at 50:1 good:bad odds, and every
+20 points doubles the odds* (PDO = 20).

```
factor = PDO / ln(2)                      = 20 / 0.6931     = 28.85
offset = 600 - factor x ln(50)            = 600 - 112.88    = 487.12
score  = offset + factor x ln(odds_good)       where odds_good = (1 - PD) / PD
points for one bin = -factor x b_j x WoE_bin  + (offset - factor x b0) / n_features
```

The intercept is spread evenly across features, so the points of all features sum **exactly** to the score.
`tests/test_scorecard.py` proves this against the formula.

**6d. Reason codes (`reason_codes`).** For each applicant, compute how many points they lost on each feature compared with the
best possible bin, and report the top 3. This is the "principal reasons for decline" a bank must give.

**6e. `decide()`.** PD below 8% means approve, 20% or more means decline, in between means refer to a human. About 48% of
out-of-time applicants are approved at this policy.

### Step 7: `train.py` (champion vs challenger)
**Three samples:**

| Sample | Rows | Purpose |
|---|---|---|
| train (75% of older data) | 11,378 | the model learns here |
| test (25% of older data) | 3,793 | unseen data from the same period |
| **out-of-time (OOT)**: newest 6 months | 4,829 | real-world check: models are always used on the *future* |

`train_test_split(..., stratify=...)` keeps the default rate equal (about 12%) in train and test.

**Challenger:** `HistGradientBoostingClassifier`, scikit-learn's fast version of the XGBoost idea (many small decision trees,
each correcting the previous ones; handles missing values natively).

**Results from this run:**

| Model | train AUC | test AUC | OOT AUC |
|---|---|---|---|
| Scorecard | 0.734 | 0.713 | **0.735** |
| Gradient boosting | 0.778 | 0.715 | 0.724 |

The challenger looks better on training data (0.778) but is *worse* on future data (0.724): it **overfitted** (memorised noise).
The simpler, explainable scorecard wins, which is exactly the argument banks use to keep scorecards. (This is the result on
*this simulator*; do not claim scorecards always beat boosting.)

**MLflow** (`log_to_mlflow`) records parameters and metrics for every run, so experiments are reproducible and comparable.
View with `mlflow ui --backend-store-uri mlruns`. Logging is wrapped in `try/except` so training still works if MLflow fails.

### Step 8: `validate.py` (the Model Risk Analyst's job)
**Principle:** the people who build a model must not be the ones who approve it. This script only *loads* the saved model and
data and **re-computes everything independently**, then writes `reports/validation_report.md`.

Sections: purpose and scope, data summary, discriminatory power (AUC/Gini/KS/Brier for both models and all samples),
calibration by risk decile, stability (PSI), fairness screening, champion vs challenger conclusion, limitations.

**Six policy checks** decide the verdict (thresholds are in `config.py`; they are *this project's* policy, not CRDB's):

| Check | Result |
|---|---|
| OOT AUC >= 0.70 | 0.735 pass |
| AUC drop train -> OOT <= 0.05 | -0.001 pass |
| Score PSI < 0.25 | 0.004 pass |
| abs(mean predicted PD - observed rate) <= 0.03 | 0.003 pass |
| Gender impact ratio >= 0.80 | 0.991 pass |
| Region impact ratio >= 0.80 | 0.922 pass |

**Verdict: "PASSED WITH CONDITIONS".** All six checks pass, but the validator found that the input
`mobile_money_monthly_txn` has drifted a lot (PSI 0.48). The final score is still stable because that feature has
modest weight, yet a good validator flags it as a *condition*. This is a realistic outcome and a strong talking point.

**Impact ratio** = lowest group approval rate / highest group approval rate. The 0.80 level echoes the "four-fifths rule" used as a
screening heuristic in some jurisdictions; it is a *screen*, not proof of fairness or a Tanzanian legal standard.

### Step 9: `monitor.py` (models decay)
Customers and the economy change, so a deployed model is re-checked monthly. The script replays the OOT months as "live data"
and reports, per month: predicted PD vs observed default rate, monthly AUC, and score PSI, plus PSI for every input.
It prints `ALERT: input 'mobile_money_monthly_txn' has drifted (PSI 0.48)` and saves `reports/monitoring.png`.
In production the same code would read the API's prediction log instead of the OOT sample.

**What you would do about the alert:** (1) check whether it is real behaviour change or a data-pipeline fault,
(2) check whether the score or approval rate moved, (3) if the drift is real and persists, refit the model on recent data.

### Step 10: `api.py` (serve the model)
Run `uvicorn mikopo.api:app --reload`, then open `http://127.0.0.1:8000/docs` for interactive documentation.

- **Pydantic `Applicant`** validates every input (age 18 to 100, allowed employment types, non-negative amounts). Bad input
  returns HTTP 422 *before* it reaches the model.
- **`Depends(get_model)`** loads the model once (`lru_cache`) and lets tests swap in a small model.
- **`/score`** computes `dti` with the *shared* function, builds a one-row DataFrame in the training column order, and returns PD,
  score, decision and the top-3 reason codes.
- **`/health`** is what Docker and load balancers ping.

Example from this run: a 29-year-old informal trader (TZS 450,000/month income, TZS 2.5M loan over 12 months, 1 past
delinquency) gets **PD 23.8%, score 520.7, decision "decline"**, with reasons `past_delinquencies`, `dti`, `age`.

### Step 11: tests, Docker, CI
- **19 tests** (`pytest`, plus 2 dashboard tests when Streamlit is installed): perfect model gives KS = Gini = 1; random model gives Gini near 0; PSI is 0 for identical data and
  flags a shift; score sums to the odds formula; ranks of PD and score are perfectly inverse; missing values do not crash; decision
  boundaries; API accepts good input and rejects bad input with 422; data-quality rules catch a corrupted row.
- **`Dockerfile`** builds an image that trains the model and serves the API (`docker run -p 8000:8000 mikopo-score`).
  *Written but not executed in the authoring environment, so test it on your machine and fix anything that breaks.*
- **`.github/workflows/ci.yml`** runs tests and the whole pipeline on every GitHub push. A green tick shows recruiters it works.

### Step 12: `dashboard/app.py` (visualisation)
Run: `pip install -e ".[dashboard]"` then `streamlit run dashboard/app.py` (opens `http://localhost:8501`).
Streamlit re-runs the whole script from top to bottom whenever you touch a widget, which is why the code reads like a simple top-to-bottom script.

| Tab | What it shows | Why it matters |
|---|---|---|
| Score an applicant | sliders and inputs, live PD / score / decision, top-3 reasons, points per feature | the demo that impresses non-technical people |
| Model performance | champion vs challenger metrics table, ROC, calibration, score distribution, the full validation report | Model Risk evidence |
| Monitoring | predicted vs observed default rate by month, input-drift bars, red ALERT boxes | MLOps evidence |
| Scorecard points | points per bin for any feature, information value per feature | explainability |
| Data warehouse | data-quality results and the SQL business query | Data Architecture evidence |

**Key lines:** `@st.cache_resource` loads the model once instead of on every click; the dashboard calls the *same*
`compute_dti()` and `decide()` as training and the API, so all three always agree; `tests/test_dashboard.py` runs the dashboard
headlessly and checks a risky applicant is declined and a safe one approved.

### Docker: how to run it
Docker packages the code, libraries and trained model into one **image**, so it runs the same on any computer.

1. Install **Docker Desktop** (Windows needs WSL 2). Start it and wait until it says it is running.
2. Open a terminal in the project root (the folder containing `Dockerfile`) and build. The first build takes several minutes, because it downloads Python, installs the libraries and runs the whole pipeline to train the model inside the image:
   `docker build -t mikopo-score .`
3. Run the API: `docker run --rm -p 8000:8000 mikopo-score` then open `http://localhost:8000/docs`.
4. Run the dashboard (second terminal): `docker run --rm -p 8501:8501 mikopo-score streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501` then open `http://localhost:8501`.
5. Stop with Ctrl+C (or `docker ps`, then `docker stop <id>`).
6. Copy reports out of the image: `docker create --name tmp mikopo-score`, then `docker cp tmp:/app/reports ./reports_from_docker`, then `docker rm tmp`.

What the flags mean: `-t mikopo-score` names the image; `-p 8000:8000` maps *your* port to the container's port (change the left number if it is taken,
for example `-p 8080:8000`); `--rm` deletes the container when it stops; `--server.address 0.0.0.0` lets Streamlit accept connections from outside the container.

Troubleshooting: *"cannot connect to the Docker daemon"* means Docker Desktop is not running. *"port is already allocated"* means change the left port number.
On PowerShell, `curl` is an alias for another command; use `curl.exe` or the `/docs` page instead. *Not tested in the authoring environment*, so if a step fails, copy the error message and fix it.

---

## 3. Defence Q&A (practise answering these out loud)

1. **Why synthetic data?** Real bank data is confidential. The simulator lets me test the whole pipeline against rules I know are true.
   Every number describes the simulator, and I say so in the report. The pipeline accepts real data with the same columns.
2. **Why a scorecard instead of XGBoost or a neural network?** Explainability (reason codes), stability, easier validation, and here it
   generalised better. I still trained gradient boosting as a challenger to *prove* the choice.
3. **What is WoE and why use it?** `ln(good% / bad%)` per bin. It linearises each feature for logistic regression, handles
   missing values and outliers, and gives a readable points table.
4. **How did you avoid data leakage?** Binning and model fitting use train only; OOT data is strictly later in time; `default` is never a feature;
   I checked IV stays below 0.5.
5. **Why an out-of-time sample?** Models are used on the future. Random splits flatter the model because they mix periods.
6. **Is AUC 0.735 good?** Credit scorecards often land roughly in the 0.65 to 0.80 range on real data, but my 0.735 only describes the simulator.
7. **Gini vs AUC?** Gini = 2 x AUC - 1. Same information, different scale.
8. **What does PSI measure, and what are the thresholds?** The shift between two distributions. Below 0.10 stable, 0.10 to 0.25 watch, above 0.25 act.
   These are widely used rules of thumb, not laws.
9. **Why is gender not a model input but still analysed?** Using it as an input is discriminatory. But removing it does not guarantee fairness because
   other variables can be proxies, so I test *outcomes* by group.
10. **Your validator flagged drift in mobile-money activity. What next?** Confirm it is real and not a pipeline fault, check score and approval
    impact, monitor monthly, and refit on recent data if it persists.
11. **Why did the challenger lose out of time?** It fit noise in the training period (train AUC 0.778, OOT 0.724). More complexity needs more data or regularisation.
12. **`dti` and `loan_term_months` are correlated. Is that a problem?** `dti` is computed from the term, so they overlap. Logistic regression coefficients can become unstable
    with correlated inputs. With real data I would check variance inflation factors or drop one.
13. **What is training-serving skew?** Features computed differently in training and production. I prevent it with the shared `features.py`.
14. **Why a star schema?** Fast, simple analytics joins; one fact table, descriptive dimensions, surrogate keys that insulate the warehouse from source-system changes.
15. **How does PD connect to bank provisioning?** Under IFRS 9, expected credit loss is roughly `PD x LGD x EAD` (loss given default, exposure at default).
    My project estimates PD only.
16. **What is reject inference?** Real data only has outcomes for approved applicants. Declined ones are unobserved, which biases the model. I list it as a limitation.
17. **What is missing before production?** Authentication, request logging, HTTPS, model governance and documentation, real data, longer outcome history, security review.
18. **What does "refer" mean?** The borderline band (PD 8% to 20%) goes to a human credit officer.
19. **Why Docker?** The same image runs identically on a laptop, a server or a cloud, so "it works on my machine" stops being an excuse. It also
    makes deployment and testing repeatable.
20. **Why a dashboard when you already have a report?** The report is the formal record for validators. The dashboard lets business users and
    non-technical audiences *interact* with the model, and shows monitoring continuously instead of once.

---

## 4. Known limitations (say them before anyone asks)

- Synthetic data; metrics describe the simulator.
- Default is a simple yes/no. Real banks define default precisely (for example 90 days past due) and need a long observation window.
- Validation thresholds are teaching policy, not CRDB's.
- No reject inference, no LGD/EAD modelling.
- One application per customer in the generated data.
- Docker and CI were not run in the authoring environment.
- MLflow uses the local-folder store for simplicity; production teams use a database backend.

---

## 5. Roadmap: from second year to the final-year project

**Now (year 2): learn it in this order.** Python and pandas, then SQL and the star schema, then logistic regression and the metrics
(AUC/KS/PSI), then WoE scorecards, then FastAPI and Git. Re-type each module yourself instead of copy-pasting, and add one small change per week.

**Year 3 upgrades (each one makes the project stronger):**
1. Replace the simulator with a public credit dataset (for example a Kaggle loan dataset) and re-run validation. Report the differences honestly.
2. Add SHAP explanations for the challenger.
3. Use PostgreSQL (Docker Compose) instead of SQLite.
4. Schedule the pipeline with Airflow; add a Streamlit monitoring dashboard.
5. Model LGD and EAD and compute IFRS 9 expected credit loss.
6. Add a model-documentation template (model card), a change log, and authentication on the API.
7. Connect it to your FinIntel idea: informal-economy lending using mobile-money signals.

**5-minute presentation outline:** problem (PD and decisions) -> data and the honesty note -> warehouse and quality checks -> scorecard and why it beat boosting
-> validation verdict and the drift condition -> API demo (`/docs`) -> limitations and roadmap.
