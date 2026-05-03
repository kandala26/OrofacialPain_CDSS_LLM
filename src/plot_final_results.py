import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = DATA_DIR / "figures_final"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

ARCH_COLORS = {
    "arch1": "#5B8E7D",
    "arch2": "#4C78A8",
    "arch3": "#E76F51",
}
CASE_COLORS = [
    "#5B8E7D",
    "#7B9ACC",
    "#A78BC0",
    "#D98C95",
    "#E6B566",
    "#8FA67A",
    "#6D7D8F",
    "#C992B0",
    "#B7A27A",
]
PLOT_BG = "#F7F4EE"
GRID = "#D6D1C8"
TEXT = "#2B2B2B"


def style_axes(ax):
    ax.set_facecolor(PLOT_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def save_fig(fig, filename):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def sort_case_frame(df):
    return df.sort_values("case_id", key=lambda s: s.str.replace("V", "", regex=False).astype(int)).reset_index(drop=True)


def human_overall_chart():
    summary = pd.read_csv(DATA_DIR / "final_results_human_overall_descriptive.csv")
    case_scores = pd.read_csv(DATA_DIR / "human_eval_latest_numeric_only.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=PLOT_BG)
    x = np.arange(len(summary))
    means = summary["mean"].values
    colors = [ARCH_COLORS[a] for a in summary["architecture"]]

    ax.bar(x, means, color=colors, width=0.6)

    for i, arch in enumerate(summary["architecture"]):
        vals = case_scores.loc[case_scores["architecture"] == arch, "mean_score"].values
        jitter = np.linspace(-0.14, 0.14, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="white", edgecolor=colors[i], s=42, zorder=3)
        ax.text(i, means[i] + 0.06, f"{means[i]:.2f}", ha="center", va="bottom", fontsize=10, color=TEXT)

    ax.set_ylim(0, 5.2)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["architecture"].str.upper())
    ax.set_ylabel("Mean expert score")
    ax.set_title("Overall Human Evaluation by Architecture", fontsize=13, weight="bold")
    style_axes(ax)
    return save_fig(fig, "human_overall_by_architecture.png")


def human_criterion_chart():
    df = pd.read_csv(DATA_DIR / "final_results_human_by_criterion.csv")
    criteria = [
        ("primary_diagnosis_accuracy", "Primary diagnosis"),
        ("differential_diagnosis_quality", "Differential diagnosis"),
        ("clinical_reasoning_quality", "Clinical reasoning"),
        ("guideline_adherence", "Guideline adherence"),
        ("overall_clinical_usefulness", "Clinical usefulness"),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor=PLOT_BG)
    x = np.arange(len(criteria))
    width = 0.22

    for idx, arch in enumerate(["arch1", "arch2", "arch3"]):
        vals = [df.loc[df["architecture"] == arch, c].iloc[0] for c, _ in criteria]
        ax.bar(x + (idx - 1) * width, vals, width=width, color=ARCH_COLORS[arch], label=arch.upper())
        for j, v in enumerate(vals):
            ax.text(x[j] + (idx - 1) * width, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT)

    ax.set_ylim(0, 5.25)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in criteria], rotation=14, ha="right")
    ax.set_ylabel("Mean expert score")
    ax.set_title("Human Evaluation by Criterion", fontsize=13, weight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    style_axes(ax)
    return save_fig(fig, "human_by_criterion.png")


def human_case_heatmap():
    df = pd.read_csv(DATA_DIR / "final_results_human_case_by_architecture.csv")
    df = sort_case_frame(df)
    matrix = df[["arch1", "arch2", "arch3"]].values

    fig, ax = plt.subplots(figsize=(5.8, 5.8), facecolor=PLOT_BG)
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=3.0, vmax=5.0, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["ARCH1", "ARCH2", "ARCH3"])
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["case_id"])
    ax.set_title("Case-by-Case Human Scores", fontsize=13, weight="bold", color=TEXT)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=TEXT, fontsize=10, weight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9, colors=TEXT)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(PLOT_BG)
    return save_fig(fig, "human_case_heatmap.png")


def human_case_group_chart():
    df = pd.read_csv(DATA_DIR / "final_results_human_by_case_group.csv")
    groups = ["Headache", "Musculoskeletal/TMD", "Neuropathic-related"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor=PLOT_BG)
    x = np.arange(len(groups))
    width = 0.22

    for idx, arch in enumerate(["arch1", "arch2", "arch3"]):
        vals = [df.loc[(df["broad_case_group"] == g) & (df["architecture"] == arch), "mean_score"].iloc[0] for g in groups]
        ax.bar(x + (idx - 1) * width, vals, width=width, color=ARCH_COLORS[arch], label=arch.upper())
        for j, v in enumerate(vals):
            ax.text(x[j] + (idx - 1) * width, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT)

    ax.set_ylim(0, 5.2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=10, ha="right")
    ax.set_ylabel("Mean expert score")
    ax.set_title("Human Evaluation by Broad Case Group (Exploratory)", fontsize=13, weight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    style_axes(ax)
    return save_fig(fig, "human_by_case_group.png")


def human_win_counts_chart():
    df = pd.read_csv(DATA_DIR / "final_results_human_win_counts.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=PLOT_BG)
    x = np.arange(len(df))
    vals = df["case_wins"].values
    colors = [ARCH_COLORS[a] for a in df["architecture"]]
    ax.bar(x, vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{int(v)}", ha="center", va="bottom", fontsize=10, color=TEXT)
    ax.set_ylim(0, max(vals) + 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["architecture"].str.upper())
    ax.set_ylabel("Number of case wins")
    ax.set_title("Human Evaluation Case Wins", fontsize=13, weight="bold")
    style_axes(ax)
    return save_fig(fig, "human_case_wins.png")


def human_mean_vs_variability_chart():
    df = pd.read_csv(DATA_DIR / "final_results_human_overall_descriptive.csv")
    fig, ax = plt.subplots(figsize=(6.8, 4.8), facecolor=PLOT_BG)
    for _, row in df.iterrows():
        x = row["mean"]
        y = row["std"]
        arch = row["architecture"]
        ax.scatter(x, y, s=240, color=ARCH_COLORS[arch], edgecolor="white", linewidth=1.6, zorder=3)
        ax.text(x + 0.015, y + 0.008, arch.upper(), fontsize=10, color=TEXT, weight="bold")
    ax.set_xlim(3.9, 4.3)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("Mean expert score")
    ax.set_ylabel("Standard deviation across cases")
    ax.set_title("Mean Performance vs Variability", fontsize=13, weight="bold")
    style_axes(ax)
    ax.grid(axis="both", color=GRID, linewidth=0.8, alpha=0.8)
    return save_fig(fig, "human_mean_vs_variability.png")


def human_average_rank_chart():
    df = pd.read_csv(DATA_DIR / "final_results_human_average_rank.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=PLOT_BG)
    x = np.arange(len(df))
    vals = df["rank_within_case"].values
    colors = [ARCH_COLORS[a] for a in df["architecture"]]
    ax.bar(x, vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=TEXT)
    ax.set_ylim(0, 3.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["architecture"].str.upper())
    ax.set_ylabel("Average within-case rank")
    ax.set_title("Average Rank by Architecture", fontsize=13, weight="bold")
    style_axes(ax)
    return save_fig(fig, "human_average_rank.png")


def rag_case_dumbbell():
    df = pd.read_csv(DATA_DIR / "final_results_arch3_rag_case_scores.csv")
    df = sort_case_frame(df)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.2, 5.4), facecolor=PLOT_BG)

    for i, row in df.iterrows():
        ax.plot([row["response_relevancy"], row["context_utilization"]], [i, i], color=GRID, linewidth=2, zorder=1)

    ax.scatter(df["response_relevancy"], y, color="#4C78A8", s=70, label="Response relevancy", zorder=3)
    ax.scatter(df["context_utilization"], y, color="#E76F51", s=70, label="Context utilization", zorder=3)

    for i, row in df.iterrows():
        ax.text(row["response_relevancy"] - 0.015, i + 0.22, f'{row["response_relevancy"]:.2f}', ha="right", va="center", fontsize=8, color=TEXT)
        ax.text(row["context_utilization"] + 0.015, i + 0.22, f'{row["context_utilization"]:.2f}', ha="left", va="center", fontsize=8, color=TEXT)

    ax.set_xlim(0, 1.02)
    ax.set_yticks(y)
    ax.set_yticklabels(df["case_id"])
    ax.invert_yaxis()
    ax.set_xlabel("Metric score")
    ax.set_title("Architecture 3 Automatic Metrics by Case", fontsize=13, weight="bold", pad=12)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    return save_fig(fig, "arch3_rag_case_dumbbell.png")


def arch3_response_relevancy_case_chart():
    df = pd.read_csv(DATA_DIR / "final_results_arch3_rag_case_scores.csv")
    df = sort_case_frame(df)
    vals = df["response_relevancy"].values
    colors = CASE_COLORS[: len(df)]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=PLOT_BG)
    ax.barh(df["case_id"], vals, color=colors)
    mean_val = vals.mean()
    ax.axvline(mean_val, linestyle="--", linewidth=1.8, color="#6A1B9A")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    ax.set_xlim(0, max(vals) + 0.12)
    ax.set_xlabel("Response relevancy")
    ax.set_ylabel("Case ID")
    ax.set_title("Architecture 3 Response Relevancy by Case", fontsize=13, weight="bold")
    ax.invert_yaxis()
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    return save_fig(fig, "arch3_response_relevancy_by_case.png")


def arch3_context_utilization_case_chart():
    df = pd.read_csv(DATA_DIR / "final_results_arch3_rag_case_scores.csv")
    df = sort_case_frame(df)
    vals = df["context_utilization"].values
    colors = CASE_COLORS[: len(df)]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=PLOT_BG)
    ax.barh(df["case_id"], vals, color=colors)
    mean_val = vals.mean()
    ax.axvline(mean_val, linestyle="--", linewidth=1.8, color="#6A1B9A")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Context utilization")
    ax.set_ylabel("Case ID")
    ax.set_title("Architecture 3 Context Utilization by Case", fontsize=13, weight="bold")
    ax.invert_yaxis()
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    return save_fig(fig, "arch3_context_utilization_by_case.png")


def arch3_rag_by_case_group_chart():
    df = pd.read_csv(DATA_DIR / "final_results_arch3_rag_by_case_group.csv")
    groups = df["broad_case_group"].tolist()
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor=PLOT_BG)
    x = np.arange(len(groups))
    width = 0.28
    rr = df["response_relevancy_mean"].values
    cu = df["context_utilization_mean"].values

    ax.bar(x - width / 2, rr, width=width, color="#4C78A8", label="Response relevancy")
    ax.bar(x + width / 2, cu, width=width, color="#E76F51", label="Context utilization")
    for i, v in enumerate(rr):
        ax.text(x[i] - width / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    for i, v in enumerate(cu):
        ax.text(x[i] + width / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT)

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=10, ha="right")
    ax.set_ylabel("Mean metric score")
    ax.set_title("Architecture 3 Automatic Metrics by Broad Case Group (Exploratory)", fontsize=13, weight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    style_axes(ax)
    return save_fig(fig, "arch3_rag_by_case_group.png")


def answer_relevancy_all_arch_chart():
    summary = pd.read_csv(DATA_DIR / "response_relevancy_all_architectures_summary.csv")
    fig, ax = plt.subplots(figsize=(6.8, 4.4), facecolor=PLOT_BG)
    x = np.arange(len(summary))
    means = summary["mean"].values
    colors = [ARCH_COLORS[a] for a in summary["architecture"]]

    ax.bar(x, means, color=colors, width=0.6)
    for i, v in enumerate(means):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=TEXT)

    ax.set_ylim(0, 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["architecture"].str.upper())
    ax.set_ylabel("Mean response relevancy")
    ax.set_title("Backup: Response Relevancy Across All Architectures", fontsize=12.5, weight="bold")
    style_axes(ax)
    return save_fig(fig, "backup_answer_relevancy_all_architectures.png")


def main():
    paths = {
        "human_overall": human_overall_chart(),
        "human_by_criterion": human_criterion_chart(),
        "human_case_heatmap": human_case_heatmap(),
        "human_by_case_group": human_case_group_chart(),
        "human_case_wins": human_win_counts_chart(),
        "human_mean_vs_variability": human_mean_vs_variability_chart(),
        "human_average_rank": human_average_rank_chart(),
        "arch3_rag_case_dumbbell": rag_case_dumbbell(),
        "arch3_response_relevancy_by_case": arch3_response_relevancy_case_chart(),
        "arch3_context_utilization_by_case": arch3_context_utilization_case_chart(),
        "arch3_rag_by_case_group": arch3_rag_by_case_group_chart(),
        "backup_answer_relevancy_all_architectures": answer_relevancy_all_arch_chart(),
    }
    print("Saved figures:")
    for label, path in paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
