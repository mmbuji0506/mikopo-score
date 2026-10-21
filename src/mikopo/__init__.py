"""MikopoScore: an end-to-end credit-risk scoring project.

"Mikopo" is Swahili for "loans". The package is split so that each file has ONE job:

    config.py     - every setting in one place (paths, seeds, cut-offs)
    features.py   - feature formulas shared by training AND the API (prevents skew)
    data_gen.py   - creates realistic synthetic loan applications
    warehouse.py  - loads the data into a star-schema database and checks its quality
    metrics.py    - AUC / Gini / KS / PSI / calibration helpers
    scorecard.py  - the WoE + logistic-regression scorecard (our "champion" model)
    train.py      - trains champion + challenger and saves them
    validate.py   - writes the independent model-validation report
    monitor.py    - drift / stability monitoring
    api.py        - FastAPI service that scores one applicant at a time
    pipeline.py   - runs everything in the correct order
"""

__version__ = "0.1.0"
