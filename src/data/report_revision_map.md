# Report Revision Map

## Use As Primary Results

- Human evaluation overall by architecture
- Human evaluation by criterion
- Case-by-case human comparison
- Architecture 3 automatic metrics:
  - response relevancy
  - context utilization

## Use As Secondary Or Exploratory Results

- Human evaluation by broad case group
- Human case wins
- Human mean vs variability
- Human average rank
- Architecture 3 automatic metrics by broad case group

## Keep Out Of The Main Report Narrative

- Faithfulness
- Correlation analyses between human and automatic metrics
- Cross-architecture answer relevancy
- Exact case-group subgroup table as a headline result
- Any "easy vs hard case" grouping that is not part of the final analysis files

## Section-By-Section Edit Guidance

### 4.4 LLM-Based CDSS Architectures

Keep this section, but tighten Architecture 3 so that it clearly says:

- Architecture 3 used the same structured prompting and reflection scaffold as Architecture 2.
- Retrieval used a summary of salient positive and abnormal findings rather than the full vignette.
- Retrieval was source-constrained, meaning evidence was retrieved separately from each curated source collection rather than only from a pooled corpus.
- The workflow should be described as a domain-specific RAG workflow, not as modular RAG.

### 4.5 Evaluation Methodology

Replace this section completely.

Delete:

- the faithfulness definition and any statement that faithfulness was part of the final evaluation
- the claim that the five evaluator domains were "primary diagnosis accuracy, differential diagnosis quality, clinical reasoning, next steps, and clinical safety"
- broad unsupported wording about qualitative review unless it is restated in a clearly bounded way

Replace with:

- 9 standardized OFP vignettes
- 3 blinded expert evaluators
- separate Qualtrics survey per vignette
- latest response retained when duplicate responses existed
- 5-point scale across the actual domains:
  - primary diagnosis accuracy
  - differential diagnosis quality
  - clinical reasoning quality
  - guideline adherence
  - overall clinical usefulness
- Architecture 3 secondary automatic evaluation using response relevancy and context utilization

### 4.6 Analytical Approach

Replace this section completely.

Delete:

- faithfulness-centered analytical framing
- any wording that implies faithfulness was the main quantitative analysis
- any leftover "next steps/usefulness" wording

Replace with:

- descriptive comparative analysis across the three architectures
- architecture-level mean scores
- criterion-level mean scores
- case-level comparison
- secondary descriptive analyses such as broad case-group patterns and variability
- Architecture 3 automatic metric summaries and case-level plots
- statement that inferential claims are limited because the dataset is small and the study is comparative/pilot in nature

### Results

Delete the entire current Results section beginning at:

- "Faithfulness Analysis of Architecture 3"

Replace with a new Results section structured as:

1. Human Evaluation Across Architectures
2. Overall Human Performance
3. Human Performance by Evaluation Criterion
4. Case-Level Human Performance
5. Exploratory Human Subgroup Findings
6. Architecture 3 Automatic Metric Results
7. Case-Level Automatic Metric Patterns

### Discussion

Delete the entire current Discussion section beginning at:

- "Interpretation of Faithfulness Findings"

Replace with a new Discussion section centered on:

- broad comparability across architectures
- Architecture 3 having the highest overall expert mean
- criterion-specific rather than universal gains
- Architecture 1 remaining competitive in guideline adherence and case wins
- Architecture 2 performing strongly on differential diagnosis quality
- Architecture 3 automatic metrics providing secondary support but not perfectly matching expert judgment
- the importance of prompt similarity between Architectures 2 and 3 when interpreting the incremental effect of retrieval

### Conclusion

Rewrite this section so that it reflects the final findings rather than faithfulness.

Focus on:

- what the study compared
- the main comparative finding
- the selective value of retrieval-grounded prompting
- the need for stronger retrieval design and broader future evaluation

## Wording Preferences To Keep Consistent

- Use "clinical reasoning quality", not "clinical reasoning"
- Use "overall clinical usefulness", not "next steps/usefulness"
- Use "RAG" after the first full-form introduction
- Define custom terms the first time they appear:
  - "source-constrained retrieval"
  - "summary-based retrieval"
- Use "exploratory subgroup analysis" rather than "exploratory human results"

## Figure Priority

### Main Report Figures

- human_overall_by_architecture.png
- human_by_criterion.png
- human_case_heatmap.png
- arch3_rag_case_dumbbell.png

### Additional Report Figures

- human_by_case_group.png
- human_case_wins.png
- human_mean_vs_variability.png
- arch3_response_relevancy_by_case.png
- arch3_context_utilization_by_case.png

### Backup Only

- human_average_rank.png
- arch3_rag_by_case_group.png
- backup_answer_relevancy_all_architectures.png
