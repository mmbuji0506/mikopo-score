import sqlite3

from mikopo.warehouse import build_warehouse, run_quality_checks


def test_clean_data_passes_all_quality_rules(small_data, tmp_path):
    db = build_warehouse(small_data, db_path=tmp_path / "wh.db")
    checks = run_quality_checks(db)
    assert (checks["status"] == "PASS").all()


def test_quality_rules_catch_bad_data(small_data, tmp_path):
    db = build_warehouse(small_data, db_path=tmp_path / "wh.db")
    conn = sqlite3.connect(db)
    # Sneak in a corrupt row: negative loan amount.
    conn.execute("UPDATE fact_loan_application SET loan_amount_tzs = -5 WHERE loan_id = 'L000001'")
    conn.commit(); conn.close()
    checks = run_quality_checks(db).set_index("rule")
    assert checks.loc["non-positive loan amount", "status"] == "FAIL"


def test_foreign_keys_link_every_fact_to_a_customer(small_data, tmp_path):
    db = build_warehouse(small_data, db_path=tmp_path / "wh.db")
    conn = sqlite3.connect(db)
    n_fact = conn.execute("SELECT COUNT(*) FROM fact_loan_application").fetchone()[0]
    n_joined = conn.execute("SELECT COUNT(*) FROM fact_loan_application f JOIN dim_customer c "
                            "ON f.customer_key = c.customer_key").fetchone()[0]
    conn.close()
    assert n_fact == n_joined == len(small_data)
