"""
File 3: run_context_utilization.py
Purpose: Run RAGAS Context Utilization (LLMContextPrecisionWithoutReference)
         on Architecture 3 outputs. No reference answer needed.

Place this file in: src/
Run from terminal:  cd src && python run_context_utilization.py
Input:              src/data/arch3_metrics_input.csv  (from make_metrics_csv.py)
Output:             src/data/arch3_context_utilization.csv

RAGAS version: 0.2.4
Evaluator LLM: gpt-4o-mini (same as faithfulness for consistency)

To use retrieval summary instead of vignette, change the line marked ← CHANGE HERE
"""

import os
import ast
import asyncio
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.llms import llm_factory

load_dotenv(find_dotenv())

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
input_csv = BASE_DIR / "data" / "arch3_metrics_input.csv"
save_path = BASE_DIR / "data" / "arch3_context_utilization.csv"

# ── Load data ─────────────────────────────────────────────────────────────
if not input_csv.exists():
    raise RuntimeError(
        f"Input CSV not found: {input_csv}\n"
        f"Run make_metrics_csv.py first."
    )

df = pd.read_csv(input_csv)
df["contexts"] = df["contexts"].apply(ast.literal_eval)

print(f"Loaded {len(df)} cases from {input_csv}")
print(f"Cases: {df['case_id'].tolist()}")

# ── Setup RAGAS ───────────────────────────────────────────────────────────
# In RAGAS 0.2.4, llm_factory uses OPENAI_API_KEY from environment automatically
llm = llm_factory("gpt-4o-mini")

scorer = LLMContextPrecisionWithoutReference(llm=llm)


# ── Score each case ───────────────────────────────────────────────────────
async def score_row(row):
    # Use combined input (same as Response Relevancy for consistency)
    combined_input = f"""Clinical vignette:

{row['question_vignette']}

Using the vignette text and the retrieved guideline excerpts, give me your diagnosis and next steps as per the instructions given to you."""

    sample = SingleTurnSample(
        user_input=combined_input,
        response=row["answer"],
        retrieved_contexts=row["contexts"],
    )
    score = await scorer.single_turn_ascore(sample)
    return score


async def main():
    scores = []

    for _, row in df.iterrows():
        print(f"Scoring {row['case_id']}...", end=" ")
        try:
            score = await score_row(row)
            scores.append(score)
            print(f"Context Utilization = {score:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            scores.append(None)

    df["context_utilization"] = scores

    # Save only the columns we need
    output_cols = ["case_id", "context_utilization", "json_file"]
    df[output_cols].to_csv(save_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────
    valid = df["context_utilization"].dropna()
    print(f"\n{'=' * 50}")
    print(f"CONTEXT UTILIZATION RESULTS")
    print(f"{'=' * 50}")
    print(f"Mean:   {valid.mean():.4f}")
    print(f"Median: {valid.median():.4f}")
    print(f"Min:    {valid.min():.4f}")
    print(f"Max:    {valid.max():.4f}")
    print(f"\nPer case:")
    print(df[["case_id", "context_utilization"]].to_string(index=False))
    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    asyncio.run(main())