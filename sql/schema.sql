-- =====================================================================
-- STAR SCHEMA for loan applications
-- One central FACT table (the events we measure) surrounded by DIMENSION
-- tables (the descriptive context: who, when, which product).
-- Why a star schema? Analysts can answer "default rate by region by quarter"
-- with simple, fast joins, and every fact row points to exactly one row in each
-- dimension. The SQL below is portable: it runs on SQLite (used here) and on
-- PostgreSQL (what a bank would run) without changes.
-- =====================================================================

DROP TABLE IF EXISTS fact_loan_application;
DROP TABLE IF EXISTS dim_customer;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_loan_product;
DROP TABLE IF EXISTS meta_load_audit;

-- WHO applied
CREATE TABLE dim_customer (
    customer_key     INTEGER PRIMARY KEY,
    customer_id      TEXT NOT NULL UNIQUE,   -- business key from the source system
    gender           TEXT NOT NULL,
    region           TEXT NOT NULL,
    employment_type  TEXT NOT NULL,
    age              INTEGER NOT NULL,
    months_with_bank INTEGER NOT NULL
);

-- WHEN they applied (one row per calendar day)
CREATE TABLE dim_date (
    date_key  INTEGER PRIMARY KEY,           -- format YYYYMMDD, e.g. 20250131
    full_date TEXT NOT NULL,
    year      INTEGER NOT NULL,
    quarter   INTEGER NOT NULL,
    month     INTEGER NOT NULL
);

-- WHICH product (here: loan term band)
CREATE TABLE dim_loan_product (
    product_key       INTEGER PRIMARY KEY,
    loan_term_months  INTEGER NOT NULL UNIQUE,
    term_band         TEXT NOT NULL          -- short / medium / long
);

-- THE MEASUREMENTS (one row per loan application)
CREATE TABLE fact_loan_application (
    loan_id                 TEXT PRIMARY KEY,
    customer_key            INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    date_key                INTEGER NOT NULL REFERENCES dim_date(date_key),
    product_key             INTEGER NOT NULL REFERENCES dim_loan_product(product_key),
    loan_amount_tzs         REAL NOT NULL,
    monthly_income_tzs      REAL,            -- NULL allowed: some applicants do not declare income
    dti                     REAL,
    existing_loans          INTEGER NOT NULL,
    past_delinquencies      INTEGER NOT NULL,
    mobile_money_txn        REAL,            -- NULL allowed: no mobile-money record
    mobile_money_avg_balance REAL,
    default_flag            INTEGER NOT NULL CHECK (default_flag IN (0, 1))
);

-- LINEAGE: which file was loaded, when, and how many rows. Auditors ask for this.
CREATE TABLE meta_load_audit (
    run_ts      TEXT NOT NULL,
    source_file TEXT NOT NULL,
    rows_loaded INTEGER NOT NULL
);

CREATE INDEX idx_fact_customer ON fact_loan_application(customer_key);
CREATE INDEX idx_fact_date     ON fact_loan_application(date_key);
