"""ETL (Extract-Transform-Load) into a star-schema warehouse + data-quality checks.

  Extract   : read data/applications.csv (the "source system")
  Transform : split the flat table into dimensions + a fact table, assign surrogate keys
  Load      : write them into a SQLite database (data/warehouse.db)
  Check     : run SQL data-quality rules; fail loudly if a critical rule is broken

SQLite stands in for PostgreSQL so the project runs on any laptop with no server.
"""
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from mikopo import config

DB_PATH = config.DATA_DIR / "warehouse.db"


def _term_band(months):
    """Group loan terms into bands: a typical product dimension attribute."""
    if months <= 6:
        return "short"
    if months <= 12:
        return "medium"
    return "long"


def build_warehouse(df, db_path=DB_PATH, source_name="applications.csv"):
    """Create the schema, then load dimensions first and the fact table last.

    Order matters: the fact table holds foreign keys, so the rows they point to
    (the dimension rows) must already exist.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")            # make SQLite enforce foreign keys
        conn.executescript((config.SQL_DIR / "schema.sql").read_text())

        # ---- dim_customer: surrogate key = 1, 2, 3 ... (independent of source IDs)
        customers = df[["customer_id", "gender", "region", "employment_type", "age", "months_with_bank"]] \
            .drop_duplicates("customer_id").reset_index(drop=True)
        customers.insert(0, "customer_key", customers.index + 1)

        # ---- dim_date: one row for EVERY day in the range, even days with no loans
        dates = pd.DataFrame({"full_date": pd.date_range(df["application_date"].min().normalize(),
                                                          df["application_date"].max().normalize())})
        dates["date_key"] = dates["full_date"].dt.strftime("%Y%m%d").astype(int)
        dates["year"] = dates["full_date"].dt.year
        dates["quarter"] = dates["full_date"].dt.quarter
        dates["month"] = dates["full_date"].dt.month
        dates["full_date"] = dates["full_date"].dt.strftime("%Y-%m-%d")

        # ---- dim_loan_product
        terms = sorted(df["loan_term_months"].unique())
        products = pd.DataFrame({"loan_term_months": terms})
        products.insert(0, "product_key", range(1, len(terms) + 1))
        products["term_band"] = products["loan_term_months"].map(_term_band)

        # ---- fact table: replace business keys with surrogate keys via merges
        fact = (df.merge(customers[["customer_id", "customer_key"]], on="customer_id")
                  .merge(products[["loan_term_months", "product_key"]], on="loan_term_months"))
        fact["date_key"] = fact["application_date"].dt.strftime("%Y%m%d").astype(int)
        fact = fact.rename(columns={
            "mobile_money_monthly_txn": "mobile_money_txn",
            "mobile_money_avg_balance_tzs": "mobile_money_avg_balance",
            config.TARGET: "default_flag",
        })[["loan_id", "customer_key", "date_key", "product_key", "loan_amount_tzs",
            "monthly_income_tzs", "dti", "existing_loans", "past_delinquencies",
            "mobile_money_txn", "mobile_money_avg_balance", "default_flag"]]

        customers.to_sql("dim_customer", conn, if_exists="append", index=False)
        dates.to_sql("dim_date", conn, if_exists="append", index=False)
        products.to_sql("dim_loan_product", conn, if_exists="append", index=False)
        fact.to_sql("fact_loan_application", conn, if_exists="append", index=False)
        conn.execute("INSERT INTO meta_load_audit VALUES (?, ?, ?)",
                     (datetime.now(timezone.utc).isoformat(timespec="seconds"), source_name, len(fact)))
        conn.commit()
    finally:
        conn.close()
    return db_path


# Each rule is (name, severity, SQL that returns the NUMBER OF VIOLATING ROWS).
# A good rule is: 0 violations = pass.
QUALITY_RULES = [
    ("duplicate loan_id", "critical",
     "SELECT COUNT(*) - COUNT(DISTINCT loan_id) FROM fact_loan_application"),
    ("orphan customer_key (fact points to missing customer)", "critical",
     "SELECT COUNT(*) FROM fact_loan_application f LEFT JOIN dim_customer c "
     "ON f.customer_key = c.customer_key WHERE c.customer_key IS NULL"),
    ("orphan date_key", "critical",
     "SELECT COUNT(*) FROM fact_loan_application f LEFT JOIN dim_date d "
     "ON f.date_key = d.date_key WHERE d.date_key IS NULL"),
    ("non-positive loan amount", "critical",
     "SELECT COUNT(*) FROM fact_loan_application WHERE loan_amount_tzs <= 0"),
    ("applicant outside age 18-65", "critical",
     "SELECT COUNT(*) FROM dim_customer WHERE age < 18 OR age > 65"),
    ("negative income", "critical",
     "SELECT COUNT(*) FROM fact_loan_application WHERE monthly_income_tzs < 0"),
    ("dti above 5 (implausible)", "warning",
     "SELECT COUNT(*) FROM fact_loan_application WHERE dti > 5"),
    ("more than 5% of income missing", "warning",
     "SELECT CASE WHEN AVG(monthly_income_tzs IS NULL) > 0.05 THEN 1 ELSE 0 END FROM fact_loan_application"),
    ("row count differs from audit log", "critical",
     "SELECT ABS((SELECT COUNT(*) FROM fact_loan_application) - (SELECT SUM(rows_loaded) FROM meta_load_audit))"),
]


def run_quality_checks(db_path=DB_PATH):
    """Execute every rule and return a results table."""
    conn = sqlite3.connect(db_path)
    try:
        rows = []
        for name, severity, sql in QUALITY_RULES:
            violations = int(conn.execute(sql).fetchone()[0] or 0)
            rows.append({"rule": name, "severity": severity, "violations": violations,
                         "status": "PASS" if violations == 0 else "FAIL"})
        return pd.DataFrame(rows)
    finally:
        conn.close()


# An example of WHY we built a star schema: a business question in plain SQL.
EXAMPLE_QUERY = """
SELECT c.employment_type,
       p.term_band,
       COUNT(*)                          AS applications,
       ROUND(AVG(f.default_flag) * 100, 1) AS default_rate_pct,
       ROUND(AVG(f.loan_amount_tzs))     AS avg_loan_tzs
FROM fact_loan_application f
JOIN dim_customer     c ON f.customer_key = c.customer_key
JOIN dim_loan_product p ON f.product_key  = p.product_key
GROUP BY c.employment_type, p.term_band
ORDER BY default_rate_pct DESC
"""


def main():
    df = pd.read_csv(config.DATA_DIR / "applications.csv", parse_dates=["application_date"])
    build_warehouse(df)
    print(f"[warehouse] star schema built at {DB_PATH}")

    checks = run_quality_checks()
    config.REPORT_DIR.mkdir(exist_ok=True)
    checks.to_csv(config.REPORT_DIR / "data_quality.csv", index=False)
    print(checks.to_string(index=False))
    failed_critical = checks[(checks["severity"] == "critical") & (checks["status"] == "FAIL")]
    if not failed_critical.empty:
        raise SystemExit("[warehouse] CRITICAL data-quality rule failed - stopping the pipeline.")

    conn = sqlite3.connect(DB_PATH)
    pd.read_sql_query(EXAMPLE_QUERY, conn).to_csv(config.REPORT_DIR / "example_query.csv", index=False)
    conn.close()
    print("[warehouse] example analytics query saved to reports/example_query.csv")


if __name__ == "__main__":
    main()
