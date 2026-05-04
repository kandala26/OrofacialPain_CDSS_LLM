# Comparing Prompting and Retrieval Strategies in an LLM-Based Clinical Decision Support System for Orofacial Pain

This repository contains the code, data artifacts, and analysis outputs for a capstone project comparing three clinician-facing large language model (LLM) workflows for orofacial pain diagnosis.

The project asks a simple but important question:

> If the base model is held constant, does changing the workflow around it improve clinician-facing diagnostic support in a complex domain with overlapping symptom presentations?

The three compared workflows use the same base generation model, `GPT-5.4`, and differ only in prompting and retrieval design.

## Project Summary

Orofacial pain is diagnostically challenging because musculoskeletal, headache-related, neuropathic, and dental conditions can present with overlapping symptoms. This project compares three LLM-based clinical decision support system (CDSS) architectures:

1. `Architecture 1`: baseline structured diagnostic prompt
2. `Architecture 2`: structured prompting with reflection
3. `Architecture 3`: structured prompting with reflection plus retrieval-augmented generation (RAG)

Nine standardized clinical vignettes were evaluated using blinded expert review across five criteria:

- primary diagnosis accuracy
- differential diagnosis quality
- clinical reasoning quality
- guideline adherence
- overall clinical usefulness

Human evaluation is the primary outcome. Automatic metrics are used only as secondary characterization of the RAG workflow in Architecture 3.

## Main Findings

Overall expert-rated mean scores:

- `Architecture 3`: `4.18 / 5`
- `Architecture 1`: `4.13 / 5`
- `Architecture 2`: `4.07 / 5`

Criterion-level pattern:

- `Architecture 2` performed best on differential diagnosis quality
- `Architecture 3` performed best on clinical reasoning quality
- `Architecture 1` performed best on guideline adherence
- `Architecture 1` also achieved the highest score in the greatest number of individual cases

Architecture 3 automatic metrics:

- response relevancy: `0.48`
- context utilization: `0.91`

Interpretation:

- the three workflows were broadly comparable overall
- different architectures showed different strength profiles
- greater architectural complexity did not automatically produce uniformly better performance
- retrieval was used meaningfully, but its effect was selective rather than universal in this implementation

## Repository Structure

```text
.
├── README.md
├── requirements.txt
└── src
    ├── arch1_baseline.py
    ├── arch2_structured.py
    ├── arch3_rag.py
    ├── build_human_eval_latest.py
    ├── analyze_final_results.py
    ├── plot_final_results.py
    ├── plot_poster_figures.py
    ├── run_response_relevancy.py
    ├── run_context_utilization.py
    ├── data
    │   ├── corpus
    │   ├── figures_final
    │   ├── outputs
    │   └── vignettes
    └── chroma
```

## Architecture Definitions

### Architecture 1: Baseline Structured Prompt

Architecture 1 uses a single-pass structured diagnostic prompt. It requests:

- primary diagnosis
- differential diagnoses
- clinical reasoning
- next steps

It does **not** use:

- reflection
- external retrieval

This serves as the baseline workflow.

### Architecture 2: Structured Prompting + Reflection

Architecture 2 uses a stronger reasoning-oriented prompt scaffold and a reflection step.

Workflow:

1. generate an initial structured diagnostic response
2. review the response for unsupported reasoning, inconsistency, or clarity problems
3. return a minimally revised final version

This architecture tests whether structured reasoning and self-critique improve output quality without retrieval.

### Architecture 3: Structured Prompting + Reflection + RAG

Architecture 3 uses the same prompting and reflection scaffold as Architecture 2, but adds retrieval-augmented generation (RAG).

Its workflow includes:

1. query reformulation from the vignette
2. semantic retrieval from curated guideline sources
3. grounded initial generation
4. reflection-based revision

This architecture tests whether adding retrieval-grounding changes the quality profile of the output.

## RAG Workflow Used in Architecture 3

This project uses a focused, domain-specific first-pass RAG workflow rather than a fully optimized advanced RAG system.

### Corpus

The retrieval corpus is curated from:

- `DC/TMD`
- `ICHD-3`
- selective `ICOP`

### Chunking

The corpus is chunked using a custom Python function in `src/arch3_rag.py`:

- fixed-size chunking
- approximately `1500` characters per chunk
- `300` character overlap

No LangChain text splitter or external chunking framework is used.

### Embeddings

Embeddings are generated using:

- `text-embedding-3-small`

This model is used for:

- document chunk embeddings
- retrieval query embeddings

### Vector Database

Embeddings are stored in:

- `Chroma`

### Query Reformulation

The full vignette is **not** used directly as the retrieval query.

Instead, the system first generates a concise summary of:

- positive findings
- abnormal findings
- diagnostically salient symptoms

This reformulated query is used to reduce retrieval noise while preserving the full vignette for final generation.

### Source-Constrained Retrieval

Retrieval is performed separately across source collections rather than as one fully pooled search:

- `3` chunks from `DC/TMD`
- `3` chunks from `ICHD-3`
- `3` chunks from `ICOP`

This creates a fixed per-source retrieval design with a maximum of `9` retrieved chunks per case.

### Reflection

After the initial grounded response is generated, a reflection step checks:

- diagnosis correctness
- unsupported claims
- inappropriate use of retrieved text
- internal inconsistency
- unnecessary or unsafe next steps

## What This Project Does Not Use

This RAG workflow does **not** include:

- reranking
- hybrid lexical + vector retrieval
- adaptive top-k retrieval
- adaptive source routing
- section-aware chunking
- claim-level citation grounding
- explicit retrieval confidence estimation

These are important future directions, but they were intentionally left out so the project could focus on workflow comparison rather than full retrieval optimization.

## Why a General-Purpose Model Was Used

The same general-purpose model was kept constant across all three workflows so that the study would isolate workflow effects rather than model-family differences.

This project does **not** compare:

- general-purpose LLMs versus medical LLMs
- different base model families
- different embedding model families

That was a deliberate design choice. Medical-domain LLMs may perform differently because of prior domain-specific pretraining, which would make it harder to determine whether improvements came from the workflow or from the model itself.

## Evaluation Design

### Human Evaluation

Outputs were evaluated by `3` blinded clinician reviewers using separate Qualtrics surveys for each vignette.

Each output was scored on a `5-point` scale across:

- primary diagnosis accuracy
- differential diagnosis quality
- clinical reasoning quality
- guideline adherence
- overall clinical usefulness

### Automatic Evaluation

Architecture 3 was also evaluated using RAGAS-based automatic metrics:

- `Response Relevancy`
- `LLMContextPrecisionWithoutReference` (reported in the project as context utilization)

These metrics are used only to characterize the retrieval-grounded workflow and do not replace clinician judgment.

## Key Scripts

### Core architecture scripts

- [src/arch1_baseline.py](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/arch1_baseline.py)
- [src/arch2_structured.py](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/arch2_structured.py)
- [src/arch3_rag.py](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/arch3_rag.py)

### Human evaluation processing

- [src/build_human_eval_latest.py](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/build_human_eval_latest.py)
- [src/analyze_final_results.py](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/analyze_final_results.py)

### Automatic metric scripts

- [src/run_response_relevancy.py](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/run_response_relevancy.py)
- [src/run_context_utilization.py](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/run_context_utilization.py)

### Plotting

- [src/plot_final_results.py](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/plot_final_results.py)
- [src/plot_poster_figures.py](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/plot_poster_figures.py)

## Important Data and Output Locations

### Vignettes

- [src/data/vignettes](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/data/vignettes)

### Curated corpus

- [src/data/corpus](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/data/corpus)

### Saved model outputs

- [src/data/outputs](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/data/outputs)

### Final analysis tables

- [src/data/final_results_human_by_criterion.csv](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/data/final_results_human_by_criterion.csv)
- [src/data/final_results_human_overall_descriptive.csv](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/data/final_results_human_overall_descriptive.csv)
- [src/data/final_results_human_win_counts.csv](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/data/final_results_human_win_counts.csv)
- [src/data/final_results_arch3_rag_overall_summary.csv](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/data/final_results_arch3_rag_overall_summary.csv)

### Final figures

- [src/data/figures_final](/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/data/figures_final)

## Setup Notes

This repository expects:

- Python
- an `OPENAI_API_KEY` in a local `.env` file or environment

The current `requirements.txt` covers only a subset of the libraries used in the full project scripts. Depending on which scripts you run, you may also need commonly used analysis and retrieval packages such as:

- `pandas`
- `matplotlib`
- `chromadb`
- `ragas`

## How to Run the Main Pieces

From the repository root:

```bash
cd src
python arch1_baseline.py
python arch2_structured.py
python arch3_rag.py
```

For analysis:

```bash
cd src
python build_human_eval_latest.py
python analyze_final_results.py
python plot_final_results.py
python plot_poster_figures.py
python run_response_relevancy.py
python run_context_utilization.py
```

## Limitations

This repository reflects a capstone-scale comparative study rather than a production-ready clinical system.

Key limitations include:

- small vignette set
- curated rather than comprehensive corpus
- selective ICOP coverage
- no reranking or advanced retrieval optimization
- no explicit claim-level citation grounding
- very similar prompting scaffolds in Architectures 2 and 3, which limits clean isolation of retrieval effects

## Bottom Line

The main takeaway from this project is:

> Holding the base model constant, changing the workflow changed the pattern of strengths more than it changed the overall ranking.

Architecture 3 performed best overall, but only modestly. The more important result was that different workflows supported different kinds of clinician-facing value.
