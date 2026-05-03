import asyncio
import json
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from ragas import SingleTurnSample
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import ResponseRelevancy


load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
VIGNETTES_DIR = BASE_DIR / "data" / "vignettes"
DATA_DIR = BASE_DIR / "data"


def load_vignette(case_id: str) -> str:
    path = VIGNETTES_DIR / f"{case_id.lower()}.md"
    return path.read_text(encoding="utf-8").strip()


def latest_output_files() -> list[Path]:
    latest: dict[tuple[str, str], tuple[str, Path]] = {}
    for path in OUTPUTS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        case_id = data.get("case_id")
        architecture = data.get("architecture")
        timestamp = data.get("timestamp", "")
        if not case_id or not architecture:
            continue

        key = (case_id, architecture)
        if key not in latest or timestamp > latest[key][0]:
            latest[key] = (timestamp, path)

    return [latest[key][1] for key in sorted(latest)]


def build_user_input(case_id: str, architecture: str) -> str:
    vignette_text = load_vignette(case_id)

    if architecture == "arch1":
        return f"""Clinical vignette:

{vignette_text}

Using only the clinical information provided in the vignette above, apply your diagnostic knowledge to interpret the findings and provide the structured assessment as instructed. Do not introduce symptoms, history, imaging, or examination findings that are not stated in the vignette"""

    if architecture == "arch2":
        return f"""Clinical vignette:

{vignette_text}

Using the vignette text and your expert knowledge, give me your diagnosis and next steps as per the instructions given to you."""

    if architecture == "arch3":
        return f"""Clinical vignette:

{vignette_text}

Using the vignette text and the retrieved guideline excerpts, give me your diagnosis and next steps as per the instructions given to you."""

    raise ValueError(f"Unknown architecture: {architecture}")


def build_response(data: dict) -> str:
    if data["architecture"] == "arch1":
        return data["response"]
    return data["final_output"]


def build_input_dataframe() -> pd.DataFrame:
    rows = []
    for path in latest_output_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        case_id = data["case_id"]
        architecture = data["architecture"]
        rows.append(
            {
                "case_id": case_id,
                "architecture": architecture,
                "json_file": path.name,
                "user_input": build_user_input(case_id, architecture),
                "answer": build_response(data),
            }
        )

    df = pd.DataFrame(rows)
    sort_order = {"arch1": 1, "arch2": 2, "arch3": 3}
    df["case_num"] = df["case_id"].str.replace("V", "", regex=False).astype(int)
    df["arch_num"] = df["architecture"].map(sort_order)
    df = df.sort_values(["case_num", "arch_num"]).drop(columns=["case_num", "arch_num"]).reset_index(drop=True)
    return df


async def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    llm = llm_factory("gpt-4o-mini")
    embeddings = embedding_factory("text-embedding-3-small")
    scorer = ResponseRelevancy(llm=llm, embeddings=embeddings)

    scores = []
    for _, row in df.iterrows():
        sample = SingleTurnSample(
            user_input=row["user_input"],
            response=row["answer"],
        )
        score = await scorer.single_turn_ascore(sample)
        scores.append(score)
        print(f'{row["case_id"]} {row["architecture"]}: {score:.4f}')

    scored = df.copy()
    scored["response_relevancy"] = scores
    return scored


async def main() -> None:
    df = build_input_dataframe()
    scored = await score_dataframe(df)

    detailed_path = DATA_DIR / "response_relevancy_all_architectures_detailed.csv"
    summary_path = DATA_DIR / "response_relevancy_all_architectures_summary.csv"

    scored[["case_id", "architecture", "response_relevancy", "json_file"]].to_csv(detailed_path, index=False)

    summary = (
        scored.groupby("architecture")["response_relevancy"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values("architecture")
        .reset_index(drop=True)
    )
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved detailed results to: {detailed_path}")
    print(f"Saved summary results to: {summary_path}")
    print("\nArchitecture summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
