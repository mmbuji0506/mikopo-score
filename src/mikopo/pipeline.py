"""Run the whole project in the correct order with one command:

    python -m mikopo.pipeline

generate data -> build warehouse + quality checks -> train -> validate -> monitor

Each step is a separate module so you can also run (and explain) them one at a time.
"""
from mikopo import data_gen, monitor, train, validate, warehouse


def main():
    steps = [
        ("1/5 Generate synthetic applications", data_gen.main),
        ("2/5 Build star-schema warehouse and run data-quality checks", warehouse.main),
        ("3/5 Train champion scorecard and challenger", train.main),
        ("4/5 Write independent validation report", validate.main),
        ("5/5 Run drift monitoring", monitor.main),
    ]
    for title, fn in steps:
        print(f"\n===== {title} =====")
        fn()
    print("\nDone. Open reports/validation_report.md, then run: uvicorn mikopo.api:app --reload")


if __name__ == "__main__":
    main()
