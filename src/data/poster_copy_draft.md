# Poster Copy Draft

## Title

Comparing Prompting and Retrieval Strategies in an LLM-Based Clinical Decision Support System for Orofacial Pain

## Authors / Affiliation

Moukthika Reddy  
Indiana University Luddy School of Informatics, Computing, and Engineering

## Summary / Abstract

Orofacial pain is diagnostically complex because symptom presentations often overlap across headache, musculoskeletal, and neuropathic conditions. This project compared three LLM-based clinical decision support system (CDSS) architectures: a baseline prompt-only model, a structured prompting model with reflection, and a structured prompting model combined with retrieval-augmented generation (RAG). Nine standardized clinical vignettes were evaluated using blinded expert review across five domains: primary diagnosis accuracy, differential diagnosis quality, clinical reasoning quality, guideline adherence, and overall clinical usefulness. The architectures were broadly comparable overall, but the hybrid RAG condition achieved the highest mean expert score and showed a different pattern of strengths from the other designs.

## Problem Addressed

Accurate diagnosis of orofacial pain is difficult because different disorders may present with similar symptoms, especially during early clinical assessment. Traditional rule-based CDSSs are often too rigid to handle incomplete or overlapping presentations. LLMs offer a more flexible approach, but their performance depends strongly on prompt design and, when used, retrieval quality.

## Project Objectives

- Compare three LLM-based CDSS architectures under the same orofacial pain vignette set.
- Evaluate whether structured prompting and RAG change expert-rated output quality.
- Identify which aspects of performance improve and which remain unchanged across architectures.

## Methodology

Nine standardized clinical vignettes were selected to represent major diagnostic categories in orofacial pain. The same base model was used across all three architectures so that the comparison reflected workflow design rather than model changes. Architecture 1 used a baseline structured diagnostic prompt. Architecture 2 used structured prompting with a reflection step. Architecture 3 used the same prompting scaffold as Architecture 2 but added RAG with curated clinical sources.

For the retrieval step, a concise summary of salient positive and abnormal findings was generated from each vignette and used as the retrieval query. This was introduced during iterative development to reduce retrieval noise from full narrative vignettes. Retrieval was source-constrained, meaning that evidence was retrieved separately from each curated source collection rather than only from a pooled corpus.

Model outputs were evaluated by three blinded clinical experts using Qualtrics. Outputs were scored on a 5-point scale across five domains: primary diagnosis accuracy, differential diagnosis quality, clinical reasoning quality, guideline adherence, and overall clinical usefulness.

## Tools Used

- GPT-5.4
- OpenAI embeddings
- Chroma vector store
- Curated DC/TMD, ICHD-3, and selective ICOP content
- Python, RAGAS, Matplotlib, Qualtrics

## Main Findings

- The three architectures were broadly comparable overall.
- Architecture 3 achieved the highest overall mean expert score.
- Architecture 2 performed best on differential diagnosis quality.
- Architecture 3 performed best on clinical reasoning quality.
- Architecture 1 remained competitive, particularly in guideline adherence and in several individual cases.

## Secondary Automatic-Metric Findings

For Architecture 3, context utilization was high overall, suggesting effective use of retrieved material. Response relevancy was moderate. These automatic metrics did not align perfectly with expert evaluation, which suggests that clinician ratings and automatic scores captured different aspects of output quality.

## Interpretation

The results suggest that structured prompting and retrieval may improve some aspects of output quality, but their effects are selective rather than uniform. The hybrid architecture showed the strongest overall expert-rated performance, but it did not dominate every evaluation domain. This indicates that architecture choice changes the pattern of strengths in a clinician-facing CDSS rather than producing a simple universal winner.

## Limitations

- The vignette set was small and intended for comparative evaluation rather than definitive benchmarking.
- Expert review involved three clinicians with related but not identical training backgrounds.
- The RAG corpus was curated and targeted rather than comprehensive, with selective ICOP coverage.
- The retrieval workflow did not include reranking, adaptive source routing, or explicit claim-level citation grounding.
- Because Architectures 2 and 3 used highly similar prompting scaffolds, prompt-related strengths and weaknesses may have carried from the structured condition into the hybrid condition.

## Future Work

- Expand corpus coverage, especially for neuropathic and mixed presentations.
- Improve chunking and retrieval design through section-aware chunking and reranking.
- Test architectures that separate prompting effects from retrieval effects more clearly.
- Evaluate larger case sets and more balanced repeated expert ratings.

## Figure Captions

### Figure 1
Overall human evaluation by architecture. Architecture 3 achieved the highest mean expert score, although differences across architectures were modest.

### Figure 2
Human evaluation by criterion. The three architectures showed different strengths across evaluation domains.

### Figure 3
Case-by-case human scores. Performance varied by case, indicating that architecture effects were not uniform across presentations.

### Figure 4
Architecture 3 automatic metrics by case. Context utilization was generally high, while response relevancy was more moderate and variable.
