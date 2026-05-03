"""
Quick test: Run faithfulness with combined vignette + user prompt
to see if it changes the zero scores for neuropathic cases.
"""

import ast
import asyncio
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import llm_factory

load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
input_csv = BASE_DIR / "data" / "arch3_metrics_input.csv"

llm = llm_factory("gpt-4o-mini")
scorer = Faithfulness(llm=llm)

async def score_row(row):
    # Use combined input (same as Response Relevancy)
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
    # Load data inside main to avoid scoping issues
    df_input = pd.read_csv(input_csv)
    df_input["contexts"] = df_input["contexts"].apply(ast.literal_eval)

    scores = []

    for _, row in df_input.iterrows():
        print(f"Scoring {row['case_id']}...", end=" ")
        score = await score_row(row)
        scores.append(score)
        print(f"Faithfulness = {score:.4f}")

    df_input["faithfulness_combined"] = scores

    # Compare with original
    old = pd.read_csv(BASE_DIR / "data" / "arch3_faithfulness_results_final.csv")
    df_comparison = df_input.merge(old[["case_id", "faithfulness"]], on="case_id", how="left")
    df_comparison = df_comparison.rename(columns={"faithfulness": "faithfulness_original"})

    print("\n" + "="*60)
    print("COMPARISON: Original vs Combined Input")
    print("="*60)
    print(df_comparison[["case_id", "faithfulness_original", "faithfulness_combined"]].to_string(index=False))

    print(f"\nOriginal mean:  {df_comparison['faithfulness_original'].mean():.4f}")
    print(f"Combined mean:  {df_comparison['faithfulness_combined'].mean():.4f}")
    print(f"Difference:     {df_comparison['faithfulness_combined'].mean() - df_comparison['faithfulness_original'].mean():.4f}")

if __name__ == "__main__":
    asyncio.run(main())