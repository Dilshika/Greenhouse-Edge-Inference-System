
import json
import os
import argparse
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.table import Table, TableStyleInfo

MODELS = ["tinyml", "xgboost", "lstm", "gru"]
TARGETS = ["Temperature", "Humidity", "CO2"]

METRICS = [
    "mae",
    "rmse",
    "r2",
    "persistence_mae",
    "persistence_rmse",
    "improvement_over_persistence_mae",
    "improvement_over_persistence_rmse",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edition", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="Training_Data_Results.xlsx")
    args = parser.parse_args()

    if os.path.exists(args.output):
        wb = load_workbook(args.output)
    else:
        wb = Workbook()

    sheet_name = "Detailed Metrics"

    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
    else:
        ws = wb.create_sheet(sheet_name)

    headers = [
        "Dataset", "Outer Split", "Inner Split",
        "Model", "Target",
        "MAE", "RMSE", "R2",
        "Persistence MAE", "Persistence RMSE",
        "Improvement MAE", "Improvement RMSE",
        "Lookback", "Horizon", "Interval Minutes",
        "Epochs Completed", "Best Validation Loss",
        "Train Windows", "Validation Windows", "Test Windows"
    ]

    if ws.max_row == 1 and ws["A1"].value is None:
        ws.append(headers)
    elif ws["A1"].value != "Dataset":
        ws.insert_rows(1)
        ws.append(headers)

    # Existing records indexed to avoid duplicates.
    existing = {}
    for row in range(2, ws.max_row + 1):
        key = tuple(ws.cell(row, col).value for col in range(1, 6))
        if all(v is not None for v in key):
            existing[key] = row

    added = 0
    updated = 0

    for outer in range(1, 7):
        for inner in range(1, 4):
            outer_id = f"{outer:02d}"
            inner_id = f"{inner:02d}"

            for model in MODELS:
                json_path = os.path.join(
                    args.results_dir,
                    args.edition,
                    f"outer_{outer_id}",
                    f"inner_{inner_id}",
                    model,
                    f"{model}_test_metrics.json"
                )

                if not os.path.exists(json_path):
                    print(f"SKIP: {json_path}")
                    continue

                with open(json_path, "r") as f:
                    data = json.load(f)

                test_metrics = (
                    data.get("test_metrics")
                    or data.get("floating_point_test_metrics")
                    or {}
                )

                for target, metrics in test_metrics.items():
                    key = (
                        args.edition,
                        f"Outer {outer_id}",
                        f"Inner {inner_id}",
                        model.upper(),
                        target
                    )

                    values = [
                        *key,
                        *[metrics.get(m) for m in METRICS],
                        data.get("lookback"),
                        data.get("horizon"),
                        data.get("interval_minutes"),
                        data.get("epochs_completed"),
                        data.get("best_validation_loss"),
                        data.get("window_counts", {}).get("train"),
                        data.get("window_counts", {}).get("validation"),
                        data.get("window_counts", {}).get("test"),
                    ]

                    if key in existing:
                        row = existing[key]
                        for col, value in enumerate(values, 1):
                            ws.cell(row, col).value = value
                        updated += 1
                    else:
                        ws.append(values)
                        existing[key] = ws.max_row
                        added += 1

    # Format worksheet
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for col in ws.columns:
        letter = col[0].column_letter
        ws.column_dimensions[letter].width = min(
            max(max(len(str(c.value or "")) for c in col) + 3, 14),
            32
        )

    for row in ws.iter_rows(min_row=2, min_col=6, max_col=12):
        for cell in row:
            cell.number_format = "0.000000"

    wb.save(args.output)

    print("\n========== RESULTS SUMMARY ==========")
    print(f"New records: {added}")
    print(f"Updated records: {updated}")
    print(f"Workbook: {args.output}")
    print("=====================================")

if __name__ == "__main__":
    main()
