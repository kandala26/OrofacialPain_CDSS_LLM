import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------
# 1. Final case-level dataset
# -----------------------------------
data = [
    {
        "case_id": "V1",
        "faithfulness": 0.5079365079365079,
        "suggested_primary_label": "Migraine",
        "broad_topic_group": "Headache",
        "difficulty": "Easy",
        "analysis_group": "Headache / Neurovascular",
    },
    {
        "case_id": "V2",
        "faithfulness": 0.8955223880597015,
        "suggested_primary_label": "Cluster headache",
        "broad_topic_group": "Headache",
        "difficulty": "Easy",
        "analysis_group": "Headache / Neurovascular",
    },
    {
        "case_id": "V3",
        "faithfulness": 0.7945205479452054,
        "suggested_primary_label": "Masticatory/cervical myofascial pain; TMJ arthralgia; bilateral TMJ disc displacement with reduction",
        "broad_topic_group": "Musculoskeletal / TMD",
        "difficulty": "Hard",
        "analysis_group": "Musculoskeletal / TMD",
    },
    {
        "case_id": "V4",
        "faithfulness": 0.9054054054054054,
        "suggested_primary_label": "Masticatory/cervical myofascial pain; TMJ arthralgia; left TMJ disc displacement with reduction",
        "broad_topic_group": "Musculoskeletal / TMD",
        "difficulty": "Hard",
        "analysis_group": "Musculoskeletal / TMD",
    },
    {
        "case_id": "V5",
        "faithfulness": 0.0,
        "suggested_primary_label": "Post-procedural trigeminal neuropathy with TMD features",
        "broad_topic_group": "Neuropathic",
        "difficulty": "Hard",
        "analysis_group": "Neuropathic",
    },
    {
        "case_id": "V6",
        "faithfulness": 0.0,
        "suggested_primary_label": "Post-procedural trigeminal neuropathy",
        "broad_topic_group": "Neuropathic",
        "difficulty": "Easy",
        "analysis_group": "Neuropathic",
    },
    {
        "case_id": "V8",
        "faithfulness": 0.0,
        "suggested_primary_label": "Post-procedural trigeminal neuropathy",
        "broad_topic_group": "Neuropathic",
        "difficulty": "Hard",
        "analysis_group": "Neuropathic",
    },
    {
        "case_id": "V10",
        "faithfulness": 0.8421052631578947,
        "suggested_primary_label": "Local myalgia with referral to teeth",
        "broad_topic_group": "Musculoskeletal / TMD",
        "difficulty": "Easy",
        "analysis_group": "Musculoskeletal / TMD",
    },
    {
        "case_id": "V11",
        "faithfulness": 0.958904109589041,
        "suggested_primary_label": "Myofascial pain with referral",
        "broad_topic_group": "Musculoskeletal / TMD",
        "difficulty": "Easy",
        "analysis_group": "Musculoskeletal / TMD",
    },
]

df = pd.DataFrame(data)

# -----------------------------------
# 2. Score categories
# -----------------------------------
def score_category(x):
    if x == 0:
        return "Zero"
    elif x < 0.5:
        return "Low"
    elif x < 0.8:
        return "Moderate"
    else:
        return "High"

df["score_category"] = df["faithfulness"].apply(score_category)

# -----------------------------------
# 3. Output folder
# -----------------------------------
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

# -----------------------------------
# 4. Summary statistics
# -----------------------------------
mean_score = df["faithfulness"].mean()
median_score = df["faithfulness"].median()
min_score = df["faithfulness"].min()
max_score = df["faithfulness"].max()
std_score = df["faithfulness"].std()

ranked = df.sort_values("faithfulness", ascending=False).reset_index(drop=True)

group_summary = (
    df.groupby("analysis_group")["faithfulness"]
    .agg(["count", "mean", "median", "min", "max"])
    .reset_index()
)

group_order = ["Headache / Neurovascular", "Musculoskeletal / TMD", "Neuropathic"]
group_summary["analysis_group"] = pd.Categorical(
    group_summary["analysis_group"],
    categories=group_order,
    ordered=True,
)
group_summary = group_summary.sort_values("analysis_group")

difficulty_summary = (
    df.groupby("difficulty")["faithfulness"]
    .agg(["count", "mean", "median", "min", "max"])
    .reset_index()
)
difficulty_summary["difficulty"] = pd.Categorical(
    difficulty_summary["difficulty"],
    categories=["Easy", "Hard"],
    ordered=True,
)
difficulty_summary = difficulty_summary.sort_values("difficulty")

category_counts = (
    df["score_category"]
    .value_counts()
    .reindex(["High", "Moderate", "Low", "Zero"])
    .fillna(0)
    .astype(int)
    .reset_index()
)
category_counts.columns = ["score_category", "count"]

zero_cases = df[df["faithfulness"] == 0].copy()

# -----------------------------------
# 5. Save CSV outputs
# -----------------------------------
df.to_csv(output_dir / "faithfulness_case_level.csv", index=False)
ranked.to_csv(output_dir / "faithfulness_ranked_cases.csv", index=False)
group_summary.to_csv(output_dir / "faithfulness_by_analysis_group.csv", index=False)
difficulty_summary.to_csv(output_dir / "faithfulness_by_difficulty.csv", index=False)
category_counts.to_csv(output_dir / "faithfulness_score_categories.csv", index=False)
zero_cases.to_csv(output_dir / "faithfulness_zero_cases.csv", index=False)

# -----------------------------------
# 6. Print summaries
# -----------------------------------
print("\n=== OVERALL FAITHFULNESS SUMMARY ===")
print(f"Number of cases: {len(df)}")
print(f"Mean faithfulness: {mean_score:.3f}")
print(f"Median faithfulness: {median_score:.3f}")
print(f"Minimum faithfulness: {min_score:.3f}")
print(f"Maximum faithfulness: {max_score:.3f}")
print(f"Standard deviation: {std_score:.3f}")

print("\n=== CASE-BY-CASE RANKING ===")
print(ranked[["case_id", "faithfulness", "analysis_group", "difficulty"]].to_string(index=False))

print("\n=== SCORE CATEGORY COUNTS ===")
print(category_counts.to_string(index=False))

print("\n=== MEAN FAITHFULNESS BY ANALYSIS GROUP ===")
print(group_summary.to_string(index=False))

print("\n=== MEAN FAITHFULNESS BY DIFFICULTY ===")
print(difficulty_summary.to_string(index=False))

print("\n=== ZERO-SCORE CASES ===")
print(zero_cases[["case_id", "analysis_group", "difficulty"]].to_string(index=False))

# -----------------------------------
# 7. Colors
# -----------------------------------
highest_highlight = "#E76F51"
base_case_colors = [
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
mean_line_color = "#6A1B9A"
group_colors = ["#4C78A8", "#F58518", "#54A24B"]
difficulty_colors = ["#7B2CBF", "#C77DFF"]
donut_colors = ["#E76F51", "#F4A261", "#2A9D8F", "#BDBDBD"]

# -----------------------------------
# 8. Figure 1: case ranking
# -----------------------------------
case_colors = base_case_colors[:len(ranked)]
case_colors[0] = highest_highlight

plt.figure(figsize=(8.4, 4.9))
plt.barh(ranked["case_id"], ranked["faithfulness"], color=case_colors)
plt.axvline(mean_score, linestyle="--", linewidth=1.8, color=mean_line_color, label=f"Mean = {mean_score:.3f}")
plt.xlim(0, 1)
plt.xlabel("Faithfulness")
plt.ylabel("Case ID")
plt.title("Architecture 3 Faithfulness by Case")
for i, v in enumerate(ranked["faithfulness"]):
    plt.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=8)
plt.gca().invert_yaxis()
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(output_dir / "faithfulness_by_case.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------------
# 9. Figure 2: analysis group means
# -----------------------------------
plt.figure(figsize=(6.8, 4.5))
plt.bar(group_summary["analysis_group"], group_summary["mean"], color=group_colors)
plt.ylim(0, 1)
plt.xlabel("Analysis Group")
plt.ylabel("Mean Faithfulness")
plt.title("Mean Faithfulness by Clinical Group")
plt.xticks(rotation=12, ha="right")
for i, v in enumerate(group_summary["mean"]):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(output_dir / "faithfulness_by_analysis_group.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------------
# 10. Figure 3: difficulty means
# -----------------------------------
plt.figure(figsize=(5.1, 4.1))
plt.bar(difficulty_summary["difficulty"], difficulty_summary["mean"], color=difficulty_colors)
plt.ylim(0, 1)
plt.xlabel("Difficulty")
plt.ylabel("Mean Faithfulness")
plt.title("Mean Faithfulness by Difficulty")
for i, v in enumerate(difficulty_summary["mean"]):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(output_dir / "faithfulness_by_difficulty.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------------
# 11. Figure 4: score category donut
# -----------------------------------
counts = category_counts["count"].values
labels = category_counts["score_category"].values

plt.figure(figsize=(5.1, 5.1))
plt.pie(
    counts,
    labels=labels,
    autopct="%1.0f%%",
    startangle=90,
    colors=donut_colors,
    textprops={"fontsize": 9},
    wedgeprops={"width": 0.42},
)
plt.title("Faithfulness Score Categories")
plt.tight_layout()
plt.savefig(output_dir / "faithfulness_score_categories.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved files:")
print(" - data/faithfulness_case_level.csv")
print(" - data/faithfulness_ranked_cases.csv")
print(" - data/faithfulness_by_analysis_group.csv")
print(" - data/faithfulness_by_difficulty.csv")
print(" - data/faithfulness_score_categories.csv")
print(" - data/faithfulness_zero_cases.csv")
print(" - data/faithfulness_by_case.png")
print(" - data/faithfulness_by_analysis_group.png")
print(" - data/faithfulness_by_difficulty.png")
print(" - data/faithfulness_score_categories.png")