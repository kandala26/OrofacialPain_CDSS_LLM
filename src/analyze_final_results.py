from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HUMAN_PATH = DATA_DIR / "human_eval_latest_numeric_only.csv"
RESPONSE_RELEVANCY_PATH = DATA_DIR / "arch3_response_relevancy_COMBINED.csv"
CONTEXT_UTILIZATION_PATH = DATA_DIR / "arch3_context_utilization.csv"

BROAD_CASE_GROUP_MAP = {
    "V1": "Headache",
    "V2": "Headache",
    "V3": "Musculoskeletal/TMD",
    "V4": "Musculoskeletal/TMD",
    "V10": "Musculoskeletal/TMD",
    "V11": "Musculoskeletal/TMD",
    "V5": "Neuropathic-related",
    "V6": "Neuropathic-related",
    "V8": "Neuropathic-related",
}

EXACT_CASE_GROUP_MAP = {
    "V1": "Headache",
    "V2": "Headache",
    "V3": "TMD/TMJ",
    "V4": "TMD/TMJ",
    "V5": "Mixed neuropathic + TMD",
    "V6": "Neuropathic",
    "V8": "Neuropathic + Musculoskeletal",
    "V10": "Muscular (Myogenous TMD)",
    "V11": "Muscular (Myogenous TMD)",
}

CRITERIA = [
    "primary_diagnosis_accuracy",
    "differential_diagnosis_quality",
    "clinical_reasoning_quality",
    "guideline_adherence",
    "overall_clinical_usefulness",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    human = pd.read_csv(HUMAN_PATH)
    response_relevancy = pd.read_csv(RESPONSE_RELEVANCY_PATH)
    context_utilization = pd.read_csv(CONTEXT_UTILIZATION_PATH)

    human["broad_case_group"] = human["case_id"].map(BROAD_CASE_GROUP_MAP)
    human["exact_case_group"] = human["case_id"].map(EXACT_CASE_GROUP_MAP)
    response_relevancy["broad_case_group"] = response_relevancy["case_id"].map(BROAD_CASE_GROUP_MAP)
    response_relevancy["exact_case_group"] = response_relevancy["case_id"].map(EXACT_CASE_GROUP_MAP)
    context_utilization["broad_case_group"] = context_utilization["case_id"].map(BROAD_CASE_GROUP_MAP)
    context_utilization["exact_case_group"] = context_utilization["case_id"].map(EXACT_CASE_GROUP_MAP)

    return human, response_relevancy, context_utilization


def save_table(df: pd.DataFrame, filename: str) -> Path:
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    return path


def build_human_outputs(human: pd.DataFrame) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    overall_descriptive = (
        human.groupby("architecture")["mean_score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values("architecture")
        .reset_index(drop=True)
    )
    outputs["human_overall_descriptive"] = save_table(
        overall_descriptive,
        "final_results_human_overall_descriptive.csv",
    )

    criterion_descriptive = (
        human.groupby("architecture")[CRITERIA + ["mean_score"]]
        .mean()
        .reset_index()
        .sort_values("architecture")
        .reset_index(drop=True)
    )
    outputs["human_by_criterion"] = save_table(
        criterion_descriptive,
        "final_results_human_by_criterion.csv",
    )

    case_by_architecture = (
        human.pivot(index="case_id", columns="architecture", values="mean_score")
        .reset_index()
    )
    outputs["human_case_by_architecture"] = save_table(
        case_by_architecture,
        "final_results_human_case_by_architecture.csv",
    )

    case_ranks = human.copy()
    case_ranks["rank_within_case"] = case_ranks.groupby("case_id")["mean_score"].rank(
        ascending=False,
        method="min",
    )
    average_rank = (
        case_ranks.groupby("architecture")["rank_within_case"]
        .mean()
        .reset_index()
        .sort_values("rank_within_case")
        .reset_index(drop=True)
    )
    outputs["human_average_rank"] = save_table(
        average_rank,
        "final_results_human_average_rank.csv",
    )

    # Count ties for winners rather than breaking them arbitrarily.
    winners = []
    for case_id, group in human.groupby("case_id"):
        best_score = group["mean_score"].max()
        winning_rows = group[group["mean_score"] == best_score]
        for _, row in winning_rows.iterrows():
            winners.append(
                {
                    "case_id": case_id,
                    "architecture": row["architecture"],
                    "winning_score": row["mean_score"],
                }
            )
    winner_df = pd.DataFrame(winners)
    win_counts = (
        winner_df.groupby("architecture")
        .size()
        .reset_index(name="case_wins")
        .sort_values(["case_wins", "architecture"], ascending=[False, True])
        .reset_index(drop=True)
    )
    outputs["human_win_counts"] = save_table(
        win_counts,
        "final_results_human_win_counts.csv",
    )

    case_group_summary = (
        human.groupby(["broad_case_group", "architecture"])[CRITERIA + ["mean_score"]]
        .mean()
        .reset_index()
        .sort_values(["broad_case_group", "architecture"])
        .reset_index(drop=True)
    )
    outputs["human_by_case_group"] = save_table(
        case_group_summary,
        "final_results_human_by_case_group.csv",
    )

    exact_case_group_summary = (
        human.groupby(["exact_case_group", "architecture"])[CRITERIA + ["mean_score"]]
        .mean()
        .reset_index()
        .sort_values(["exact_case_group", "architecture"])
        .reset_index(drop=True)
    )
    outputs["human_by_exact_case_group"] = save_table(
        exact_case_group_summary,
        "final_results_human_by_exact_case_group.csv",
    )

    arch3_human = (
        human[human["architecture"] == "arch3"][
            ["case_id", "broad_case_group", "exact_case_group", "mean_score"] + CRITERIA
        ]
        .rename(columns={"mean_score": "arch3_human_mean_score"})
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    outputs["human_arch3_case_scores"] = save_table(
        arch3_human,
        "final_results_arch3_human_case_scores.csv",
    )

    return outputs


def build_rag_outputs(
    human: pd.DataFrame,
    response_relevancy: pd.DataFrame,
    context_utilization: pd.DataFrame,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    rag_case_scores = (
        response_relevancy[["case_id", "broad_case_group", "exact_case_group", "response_relevancy"]]
        .merge(
            context_utilization[["case_id", "context_utilization"]],
            on="case_id",
            how="inner",
        )
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    outputs["rag_case_scores"] = save_table(
        rag_case_scores,
        "final_results_arch3_rag_case_scores.csv",
    )

    rag_overall_summary = pd.DataFrame(
        [
            {
                "metric": "response_relevancy_combined",
                "count": response_relevancy["response_relevancy"].count(),
                "mean": response_relevancy["response_relevancy"].mean(),
                "median": response_relevancy["response_relevancy"].median(),
                "std": response_relevancy["response_relevancy"].std(),
                "min": response_relevancy["response_relevancy"].min(),
                "max": response_relevancy["response_relevancy"].max(),
            },
            {
                "metric": "context_utilization",
                "count": context_utilization["context_utilization"].count(),
                "mean": context_utilization["context_utilization"].mean(),
                "median": context_utilization["context_utilization"].median(),
                "std": context_utilization["context_utilization"].std(),
                "min": context_utilization["context_utilization"].min(),
                "max": context_utilization["context_utilization"].max(),
            },
        ]
    )
    outputs["rag_overall_summary"] = save_table(
        rag_overall_summary,
        "final_results_arch3_rag_overall_summary.csv",
    )

    rag_by_case_group = (
        rag_case_scores.groupby("broad_case_group")[["response_relevancy", "context_utilization"]]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    rag_by_case_group.columns = [
        "broad_case_group",
        "response_relevancy_count",
        "response_relevancy_mean",
        "response_relevancy_median",
        "response_relevancy_min",
        "response_relevancy_max",
        "context_utilization_count",
        "context_utilization_mean",
        "context_utilization_median",
        "context_utilization_min",
        "context_utilization_max",
    ]
    outputs["rag_by_case_group"] = save_table(
        rag_by_case_group,
        "final_results_arch3_rag_by_case_group.csv",
    )

    arch3_human = (
        human[human["architecture"] == "arch3"][
            ["case_id", "broad_case_group", "exact_case_group", "mean_score"]
        ]
        .rename(columns={"mean_score": "arch3_human_mean_score"})
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    rag_vs_human = arch3_human.merge(
        rag_case_scores,
        on=["case_id", "broad_case_group", "exact_case_group"],
        how="inner",
    )
    outputs["rag_vs_human_case_scores"] = save_table(
        rag_vs_human,
        "final_results_arch3_rag_vs_human_case_scores.csv",
    )

    correlation_summary = pd.DataFrame(
        [
            {
                "comparison": "arch3_human_mean_vs_response_relevancy",
                "pearson_r": rag_vs_human["arch3_human_mean_score"].corr(
                    rag_vs_human["response_relevancy"],
                    method="pearson",
                ),
                "spearman_rho": rag_vs_human["arch3_human_mean_score"].corr(
                    rag_vs_human["response_relevancy"],
                    method="spearman",
                ),
            },
            {
                "comparison": "arch3_human_mean_vs_context_utilization",
                "pearson_r": rag_vs_human["arch3_human_mean_score"].corr(
                    rag_vs_human["context_utilization"],
                    method="pearson",
                ),
                "spearman_rho": rag_vs_human["arch3_human_mean_score"].corr(
                    rag_vs_human["context_utilization"],
                    method="spearman",
                ),
            },
            {
                "comparison": "response_relevancy_vs_context_utilization",
                "pearson_r": rag_vs_human["response_relevancy"].corr(
                    rag_vs_human["context_utilization"],
                    method="pearson",
                ),
                "spearman_rho": rag_vs_human["response_relevancy"].corr(
                    rag_vs_human["context_utilization"],
                    method="spearman",
                ),
            },
        ]
    )
    outputs["rag_correlation_summary"] = save_table(
        correlation_summary,
        "final_results_arch3_rag_correlation_summary.csv",
    )

    return outputs


def main() -> None:
    human, response_relevancy, context_utilization = load_inputs()
    human_outputs = build_human_outputs(human)
    rag_outputs = build_rag_outputs(human, response_relevancy, context_utilization)

    print("Saved human analysis tables:")
    for label, path in human_outputs.items():
        print(f"- {label}: {path}")

    print("\nSaved RAG analysis tables:")
    for label, path in rag_outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
