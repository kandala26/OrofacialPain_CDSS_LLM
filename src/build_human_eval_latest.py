import csv
import re
from pathlib import Path
from statistics import mean

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = Path("/Users/moukthikareddy/Downloads")
OUTPUT_DIR = BASE_DIR / "data"

CSV_GLOB = "Case+V*.csv"

# Pulled from CASE DISTRUBUTION.xlsx.
CASE_ARCH_MAPPING = {
    "V1": {"A": "arch1", "B": "arch2", "C": "arch3"},
    "V2": {"A": "arch1", "B": "arch2", "C": "arch3"},
    "V3": {"A": "arch2", "B": "arch3", "C": "arch1"},
    "V4": {"A": "arch2", "B": "arch3", "C": "arch1"},
    "V5": {"A": "arch3", "B": "arch1", "C": "arch2"},
    "V6": {"A": "arch3", "B": "arch1", "C": "arch2"},
    "V8": {"A": "arch1", "B": "arch2", "C": "arch3"},
    "V10": {"A": "arch2", "B": "arch3", "C": "arch1"},
    "V11": {"A": "arch3", "B": "arch1", "C": "arch2"},
}

CRITERIA = [
    "primary_diagnosis_accuracy",
    "differential_diagnosis_quality",
    "clinical_reasoning_quality",
    "guideline_adherence",
    "overall_clinical_usefulness",
]


def extract_case_id(filename: str) -> str:
    match = re.search(r"Case\+?(V\d+)", filename)
    if not match:
        raise ValueError(f"Could not parse case ID from filename: {filename}")
    return match.group(1)


def parse_numeric_score(value: str) -> int | None:
    if not value:
        return None
    match = re.match(r"\s*(\d+)", value)
    return int(match.group(1)) if match else None


def load_latest_response(csv_path: Path) -> tuple[list[str], list[str]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if len(rows) <= 3:
        raise ValueError(f"No response rows found in {csv_path}")

    header = rows[0]
    data_rows = rows[3:]
    recorded_idx = header.index("RecordedDate")

    latest_row = max(data_rows, key=lambda row: row[recorded_idx])

    return header, latest_row


def row_to_output_records(case_id: str, header: list[str], latest_row: list[str], source_file: str) -> list[dict]:
    mapping = CASE_ARCH_MAPPING[case_id]
    output_records = []
    recorded_idx = header.index("RecordedDate")
    response_idx = header.index("ResponseId")

    for output_label, start_idx in {"A": 17, "B": 22, "C": 27}.items():
        values = latest_row[start_idx:start_idx + 5]
        numeric_scores = [parse_numeric_score(v) for v in values]

        record = {
            "case_id": case_id,
            "output_label": output_label,
            "architecture": mapping[output_label],
            "case_arch_id": f"{case_id}_{mapping[output_label]}",
            "source_file": source_file,
            "recorded_date": latest_row[recorded_idx],
            "response_id": latest_row[response_idx],
            "primary_diagnosis_accuracy_text": values[0],
            "differential_diagnosis_quality_text": values[1],
            "clinical_reasoning_quality_text": values[2],
            "guideline_adherence_text": values[3],
            "overall_clinical_usefulness_text": values[4],
            "primary_diagnosis_accuracy": numeric_scores[0],
            "differential_diagnosis_quality": numeric_scores[1],
            "clinical_reasoning_quality": numeric_scores[2],
            "guideline_adherence": numeric_scores[3],
            "overall_clinical_usefulness": numeric_scores[4],
            "mean_score": mean([score for score in numeric_scores if score is not None]),
        }
        output_records.append(record)

    return output_records


def build_master_dataframe() -> pd.DataFrame:
    records = []
    csv_files = sorted(DOWNLOADS_DIR.glob(CSV_GLOB))
    if not csv_files:
        raise RuntimeError(f"No files matched {CSV_GLOB} in {DOWNLOADS_DIR}")

    for csv_file in csv_files:
        case_id = extract_case_id(csv_file.name)
        header, latest_row = load_latest_response(csv_file)
        records.extend(row_to_output_records(case_id, header, latest_row, csv_file.name))

    df = pd.DataFrame(records)
    return df.sort_values(["case_id", "output_label"]).reset_index(drop=True)


def build_architecture_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("architecture")[CRITERIA + ["mean_score"]]
        .mean()
        .reset_index()
        .sort_values("architecture")
        .reset_index(drop=True)
    )


def build_numeric_only_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_id",
        "output_label",
        "architecture",
        "case_arch_id",
        "source_file",
        "recorded_date",
        "response_id",
        "primary_diagnosis_accuracy",
        "differential_diagnosis_quality",
        "clinical_reasoning_quality",
        "guideline_adherence",
        "overall_clinical_usefulness",
        "mean_score",
    ]
    return df[columns].copy()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master_df = build_master_dataframe()
    numeric_only_df = build_numeric_only_dataframe(master_df)
    summary_df = build_architecture_summary(master_df)

    master_path = OUTPUT_DIR / "human_eval_latest_long.csv"
    numeric_only_path = OUTPUT_DIR / "human_eval_latest_numeric_only.csv"
    summary_path = OUTPUT_DIR / "human_eval_latest_by_architecture.csv"

    master_df.to_csv(master_path, index=False)
    numeric_only_df.to_csv(numeric_only_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved long-format data to: {master_path}")
    print(f"Saved numeric-only data to: {numeric_only_path}")
    print(f"Saved architecture summary to: {summary_path}")
    print("\nArchitecture means:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
