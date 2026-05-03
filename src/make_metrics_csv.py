"""
File 1: make_metrics_csv.py
Purpose: Build a clean CSV from your Architecture 3 JSON output files.
         Includes both the full vignette text and the retrieval query summary
         so you can choose which to use as user_input for RAGAS metrics.

Place this file in: src/
Run from terminal:  cd src && python make_metrics_csv.py
Output:             src/data/arch3_metrics_input.csv
"""

import json
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent          # src/
output_dir = BASE_DIR / "data" / "outputs"
save_path  = BASE_DIR / "data" / "arch3_metrics_input.csv"

rows = []

json_files = sorted(output_dir.glob("*_arch3_*.json"))
if not json_files:
    raise RuntimeError(f"No arch3 JSON files found in {output_dir}")

for f in json_files:
    with open(f, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    case_id = data["case_id"]

    # ── Read the actual vignette text ──────────────────────────────────
    vignette_file = Path(data["vignette_file"])
    if vignette_file.exists():
        with open(vignette_file, "r", encoding="utf-8") as vf:
            vignette_text = vf.read().strip()
        print(f"[OK]      {case_id} — vignette loaded from {vignette_file.name}")
    else:
        print(f"[WARNING] {case_id} — vignette file not found: {vignette_file}")
        print(f"          Falling back to retrieval query as vignette text")
        vignette_text = data["retrieval_query_used"]

    # ── Contexts: already a Python list in the JSON ───────────────────
    contexts = data["retrieved_chunks"]   # list of 9 strings

    # ── Answer: use final_output only (same as answer, no duplicates) ─
    answer = data["final_output"]

    rows.append({
        "case_id":            case_id,
        "question_vignette":  vignette_text,                # full vignette text
        "question_retrieval": data["retrieval_query_used"],  # retrieval summary
        "contexts":           contexts,                      # list of 9 chunks
        "answer":             answer,                        # final output only
        "json_file":          f.name,
    })

# ── Sort by case_id and save ──────────────────────────────────────────────
import pandas as pd

df = pd.DataFrame(rows)

# Sort case_id naturally: V1, V2, ..., V10, V11 (not V1, V10, V11, V2...)
df["sort_key"] = df["case_id"].str.replace("V", "").astype(int)
df = df.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)

df.to_csv(save_path, index=False)

print(f"\n{'='*50}")
print(f"Saved {len(df)} rows to {save_path}")
print(f"{'='*50}")
print(f"\nCases found:")
for _, row in df.iterrows():
    print(f"  {row['case_id']} — {row['json_file']}")

print(f"\nColumns: {list(df.columns)}")
print(f"\nNext step: run  python run_response_relevancy.py")