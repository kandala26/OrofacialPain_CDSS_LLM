# Poster Story Map

## Core Story

This poster compares three LLM-based clinical decision support system (CDSS) architectures for orofacial pain: a baseline model, a structured prompting model, and a hybrid prompting-plus-RAG model. The main finding is that the three architectures were broadly comparable overall, but their strengths differed by evaluation domain. The hybrid architecture achieved the highest overall expert-rated mean score, while the baseline and structured architectures remained competitive in specific dimensions. These findings suggest that retrieval support may add value, but its effect is selective and depends on both prompt design and retrieval quality.

## What To Emphasize

- Human expert evaluation is the primary result.
- Automatic metrics are secondary and are used to characterize the RAG condition rather than replace clinician judgment.
- The study does not claim that one architecture was universally superior.
- The more interesting result is that architecture choice changed the pattern of strengths rather than producing a simple winner-takes-all result.

## Poster Layout For The Template

### Title Area

Use a concise title that highlights both the domain and the comparison:

`Comparing Prompting and Retrieval Strategies in an LLM-Based Clinical Decision Support System for Orofacial Pain`

Subtitle / author line:

`Moukthika Reddy`

`Indiana University Luddy School of Informatics, Computing, and Engineering`

### Summary / Abstract

Use 3 to 4 sentences:

- Orofacial pain is diagnostically complex because symptom presentations often overlap across musculoskeletal, neuropathic, and headache-related disorders.
- This study compared three LLM-based CDSS architectures: baseline prompting, structured prompting with reflection, and structured prompting combined with retrieval-augmented generation (RAG).
- Nine standardized clinical vignettes were evaluated using blinded expert review across five domains: primary diagnosis accuracy, differential diagnosis quality, clinical reasoning quality, guideline adherence, and overall clinical usefulness.
- The architectures were broadly comparable overall, but the hybrid RAG architecture achieved the highest mean expert score and showed a different pattern of strengths from the other designs.

### Prior Research Review

Keep this short and high-level:

- Prior OFP studies have evaluated prompt-based LLM decision support, but comparative evaluation of alternative architectures remains limited.
- RAG has been proposed as a way to improve grounding in clinician-facing AI, but its effect depends on corpus quality, retrieval design, and workflow integration.
- This project contributes a comparative OFP study across baseline, structured, and hybrid retrieval-grounded architectures under the same vignette conditions.

### Problem Addressed / Objectives

Use two blocks:

Problem:

- OFP diagnosis is challenging because symptoms overlap across multiple pain mechanisms and early presentations are often incomplete.
- Traditional rule-based CDSSs are difficult to maintain and may not handle nuanced diagnostic reasoning well.

Objectives:

- Compare three LLM-based CDSS architectures under the same OFP vignette set.
- Evaluate whether structured prompting and RAG influence expert-rated output quality.
- Identify where architecture changes improve performance and where they do not.

### Methodology / Project Execution

Keep this visual and compact:

- 9 standardized OFP vignettes
- Same base model across all conditions
- Architecture 1: baseline prompt
- Architecture 2: structured prompting plus reflection
- Architecture 3: structured prompting plus RAG
- Curated clinical sources: DC/TMD, ICHD-3, and selective ICOP content
- Blinded Qualtrics expert evaluation with 3 domain specialists
- Five scoring domains: primary diagnosis accuracy, differential diagnosis quality, clinical reasoning quality, guideline adherence, overall clinical usefulness

For the RAG description, use:

`Architecture 3 used a domain-specific RAG workflow with summary-based retrieval and source-constrained evidence retrieval from curated clinical sources.`

If space allows, add one sentence:

`Summary-based retrieval was introduced during iterative development to reduce retrieval noise from full narrative vignettes while preserving the full vignette for generation.`

### Findings / Conclusions

This is the most important panel. Build it around the figures:

Main findings bullets:

- The three architectures were broadly comparable, with Architecture 3 achieving the highest overall expert-rated mean score.
- Performance differences were criterion-specific rather than uniform.
- Architecture 2 performed best on differential diagnosis quality.
- Architecture 3 performed best on clinical reasoning quality and on the overall mean.
- Architecture 1 remained competitive and showed strong performance in guideline adherence and several individual cases.

Automatic metric bullets:

- For Architecture 3, context utilization was high overall, suggesting effective use of retrieved material.
- Response relevancy was moderate and did not perfectly align with expert evaluation, indicating that automatic metrics and clinician judgment captured different aspects of quality.

Interpretation bullets:

- Retrieval support may improve some aspects of reasoning, but its effect appears selective rather than universal.
- Similarity between the structured and hybrid prompting scaffolds may have reduced the apparent incremental effect of retrieval.

### Limitations / Future Work

Use balanced, non-defensive wording:

- The vignette set was small and intended for comparative evaluation rather than definitive benchmarking.
- Expert review involved three clinicians with different but related training backgrounds.
- The RAG corpus was curated and targeted rather than comprehensive, with selective ICOP coverage.
- The retrieval workflow did not include reranking, adaptive source routing, or explicit claim-level citation grounding.
- Future work should expand corpus coverage, strengthen retrieval design, and test architectures that separate prompting effects from retrieval effects more clearly.

### References / Acknowledgements

Keep only the highest-value sources on the poster:

- Vueghs et al. OFP GPT-4 CDSS paper
- One RAG-in-healthcare review
- DC/TMD
- ICHD-3
- ICOP

## Figure Placement Recommendation

Use four primary visuals and one optional backup:

1. `human_overall_by_architecture.png`
   - Place near the top right of Findings
   - Use as the headline figure

2. `human_by_criterion.png`
   - Place below or beside the headline figure
   - Use to support the “criterion-specific strengths” message

3. `human_case_heatmap.png`
   - Place below the criterion chart
   - Use to show that differences varied by case

4. `arch3_rag_case_dumbbell.png`
   - Place at the bottom of Findings or beside Limitations
   - Use as the secondary automatic-metrics figure

Optional backup only:

5. `backup_answer_relevancy_all_architectures.png`
   - Use only if there is room or if you want a discussion appendix
   - Do not treat it as the main result

## What To Leave Out

- Do not include faithfulness.
- Do not include correlations on the main poster.
- Do not include a dashboard or rebuilt prototype.
- Do not overload the poster with every exploratory subgroup breakdown.

