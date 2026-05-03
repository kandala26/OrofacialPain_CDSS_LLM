import os
import ast
import asyncio
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from ragas.llms import llm_factory
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found.")

df = pd.read_csv("data/arch3_faithfulness.csv")
df["contexts"] = df["contexts"].apply(ast.literal_eval)

llm = llm_factory("gpt-4o-mini")
scorer = Faithfulness(llm=llm)

async def score_row(row):
    sample = SingleTurnSample(
        user_input=row["question"],
        response=row["answer"],
        retrieved_contexts=row["contexts"]
    )
    score = await scorer.single_turn_ascore(sample)
    return score

async def main():
    scores = []
    for _, row in df.iterrows():
        score = await score_row(row)
        scores.append(score)
        print(f'{row["case_id"]}: {score}')

    df["faithfulness"] = scores
    df.to_csv("data/arch3_faithfulness_results_final.csv", index=False)

    print("\nSaved results to data/arch3_faithfulness_results_final.csv")
    print("\nMean faithfulness:", df["faithfulness"].mean())

if __name__ == "__main__":
    asyncio.run(main())