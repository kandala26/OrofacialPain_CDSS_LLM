from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = DATA_DIR / "figures_final"


ARCH_COLORS = {
    "arch1": "#7AAE7F",
    "arch2": "#8B78C6",
    "arch3": "#D8B34B",
}
TEXT = "#1F2933"
GRID = "#D9E2EC"
WHITE = "#FFFFFF"
LEGEND_FACE = "#FFFFFF"
LEGEND_EDGE = "#D8DEE8"


def save_fig(fig, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def style_axes(ax, *, grid_axis: str = "y") -> None:
    ax.set_facecolor(WHITE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#BCCCDC")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=TEXT, labelsize=12)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.9)


def sort_case_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        "case_id",
        key=lambda s: s.str.replace("V", "", regex=False).astype(int),
    ).reset_index(drop=True)


def poster_human_overall_chart() -> Path:
    summary = pd.read_csv(DATA_DIR / "final_results_human_overall_descriptive.csv")
    case_scores = pd.read_csv(DATA_DIR / "human_eval_latest_numeric_only.csv")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=WHITE)
    x = np.arange(len(summary))
    means = summary["mean"].values
    colors = [ARCH_COLORS[a] for a in summary["architecture"]]

    ax.bar(x, means, color=colors, width=0.72, edgecolor="none", zorder=2)

    for i, arch in enumerate(summary["architecture"]):
        vals = case_scores.loc[case_scores["architecture"] == arch, "mean_score"].values
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=WHITE,
            edgecolor=colors[i],
            linewidth=1.5,
            s=44,
            zorder=3,
        )
        ax.text(
            i,
            means[i] + 0.08,
            f"{means[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=TEXT,
        )

    ax.set_ylim(0, 5.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["ARCH1", "ARCH2", "ARCH3"], fontweight="bold", fontsize=13)
    ax.set_ylabel("Mean expert score", fontsize=12.5, fontweight="bold")
    ax.set_title("Overall Human Evaluation by Architecture", fontsize=15, fontweight="bold", pad=10)
    style_axes(ax, grid_axis="y")
    ax.margins(x=0.06)
    fig.tight_layout(pad=0.8)
    return save_fig(fig, "poster_human_overall_by_architecture.png")


def poster_human_criterion_chart() -> Path:
    df = pd.read_csv(DATA_DIR / "final_results_human_by_criterion.csv")
    criteria = [
        ("primary_diagnosis_accuracy", "Primary\ndiagnosis"),
        ("differential_diagnosis_quality", "Differential\ndiagnosis"),
        ("clinical_reasoning_quality", "Clinical\nreasoning"),
        ("guideline_adherence", "Guideline\nadherence"),
        ("overall_clinical_usefulness", "Clinical\nusefulness"),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.95), facecolor=WHITE)
    x = np.arange(len(criteria))
    width = 0.26

    for idx, arch in enumerate(["arch1", "arch2", "arch3"]):
        vals = [df.loc[df["architecture"] == arch, c].iloc[0] for c, _ in criteria]
        xpos = x + (idx - 1) * width
        ax.bar(
            xpos,
            vals,
            width=width,
            color=ARCH_COLORS[arch],
            edgecolor="none",
            label=arch.upper(),
            zorder=2,
        )
        for j, v in enumerate(vals):
            ax.text(
                xpos[j],
                v + 0.05,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.6,
                fontweight="bold",
                color=TEXT,
            )

    ax.set_ylim(0, 5.35)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in criteria], fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Mean expert score", fontsize=12.5, fontweight="bold")
    ax.set_title("Human Evaluation by Criterion", fontsize=15, fontweight="bold", pad=12)
    leg = ax.legend(
        frameon=True,
        fancybox=True,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=9.8,
        handlelength=1.2,
        columnspacing=1.2,
        borderpad=0.45,
        labelspacing=0.5,
    )
    leg.get_frame().set_facecolor(LEGEND_FACE)
    leg.get_frame().set_edgecolor(LEGEND_EDGE)
    leg.get_frame().set_linewidth(0.9)
    leg.get_frame().set_alpha(1.0)
    for text in leg.get_texts():
        text.set_color(TEXT)
        text.set_fontweight("bold")
    style_axes(ax, grid_axis="y")
    ax.margins(x=0.03)
    fig.subplots_adjust(top=0.90, left=0.08, right=0.98, bottom=0.16)
    return save_fig(fig, "poster_human_by_criterion.png")


def poster_rag_case_dumbbell() -> Path:
    df = pd.read_csv(DATA_DIR / "final_results_arch3_rag_case_scores.csv")
    df = sort_case_frame(df)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.9, 5.2), facecolor=WHITE)

    for i, row in df.iterrows():
        ax.plot(
            [row["response_relevancy"], row["context_utilization"]],
            [i, i],
            color="#C5D1DC",
            linewidth=2.2,
            zorder=1,
        )

    ax.scatter(
        df["response_relevancy"],
        y,
        color=ARCH_COLORS["arch2"],
        s=76,
        label="Response relevancy",
        zorder=3,
    )
    ax.scatter(
        df["context_utilization"],
        y,
        color=ARCH_COLORS["arch3"],
        s=76,
        label="Context utilization",
        zorder=3,
    )

    for i, row in df.iterrows():
        ax.text(
            row["response_relevancy"] - 0.02,
            i + 0.22,
            f'{row["response_relevancy"]:.2f}',
            ha="right",
            va="center",
            fontsize=9.6,
            fontweight="bold",
            color=TEXT,
        )
        ax.text(
            row["context_utilization"] + 0.02,
            i + 0.22,
            f'{row["context_utilization"]:.2f}',
            ha="left",
            va="center",
            fontsize=9.6,
            fontweight="bold",
            color=TEXT,
        )

    ax.set_xlim(0, 1.04)
    ax.set_yticks(y)
    ax.set_yticklabels(df["case_id"], fontweight="bold", fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xlabel("Metric score", fontsize=12.5, fontweight="bold")
    ax.set_title("Architecture 3 Automatic Metrics by Case", fontsize=15, fontweight="bold", pad=12)
    leg = ax.legend(
        frameon=True,
        fancybox=True,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        fontsize=10.2,
        handlelength=1.4,
        columnspacing=1.2,
        borderpad=0.45,
        labelspacing=0.45,
    )
    leg.get_frame().set_facecolor(LEGEND_FACE)
    leg.get_frame().set_edgecolor(LEGEND_EDGE)
    leg.get_frame().set_linewidth(0.9)
    leg.get_frame().set_alpha(1.0)
    for text in leg.get_texts():
        text.set_color(TEXT)
        text.set_fontweight("bold")
    style_axes(ax, grid_axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(top=0.90, left=0.10, right=0.98, bottom=0.12)
    return save_fig(fig, "poster_arch3_rag_case_dumbbell.png")


def main() -> None:
    paths = {
        "poster_human_overall": poster_human_overall_chart(),
        "poster_human_by_criterion": poster_human_criterion_chart(),
        "poster_arch3_rag_case_dumbbell": poster_rag_case_dumbbell(),
    }
    print("Saved poster figures:")
    for label, path in paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
