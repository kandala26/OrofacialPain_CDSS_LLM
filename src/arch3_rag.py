import os
import uuid
import json
from pathlib import Path
from datetime import datetime

import chromadb
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found.")

client = OpenAI(api_key=api_key)

model_name = "gpt-5.4"
embedding_model = "text-embedding-3-small"

BASE_DIR = Path(__file__).resolve().parent
vignette_path = BASE_DIR / "data" / "vignettes" / "v5.md"
corpus_dir = BASE_DIR / "data" / "corpus"
output_dir = BASE_DIR / "data" / "outputs"

chroma_dir = BASE_DIR / "chroma"
chroma_dir.mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(chroma_dir / "chroma_store"))
collection_name = "ofp_corpus"

REBUILD_CORPUS = True  # set to True if you add new corpus files

def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size=1500, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def embed_texts(texts, model=embedding_model):
    response = client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in response.data]

def embed_query(query, model=embedding_model):
    response = client.embeddings.create(input=[query], model=model)
    return response.data[0].embedding

def reset_and_store_corpus():
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    new_collection = chroma_client.get_or_create_collection(name=collection_name)
    md_files = sorted(corpus_dir.rglob("*.md"))
    if not md_files:
        raise RuntimeError(f"No markdown files found in corpus directory: {corpus_dir}")

    for file_path in md_files:
        text = load_text(file_path)
        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)
        parent_folder = file_path.parent.name.lower()

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            new_collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[str(uuid.uuid4())],
                metadatas=[{
                    "source": file_path.name,
                    "corpus": parent_folder,
                    "relative_path": str(file_path.relative_to(corpus_dir)),
                    "chunk_index": i,
                    "length": len(chunk),
                }]
            )

    print(f"\nCorpus loaded from: {corpus_dir}")
    print(f"Files indexed: {[f.name for f in md_files]}")
    return new_collection

def get_or_build_corpus():
    existing = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing:
        print("\nCorpus already indexed. Loading existing collection.")
        return chroma_client.get_collection(name=collection_name)
    return reset_and_store_corpus()

def summarize_vignette_for_retrieval(vignette_text: str) -> str:

    system_prompt = '''You are a clinical summarizer. Your task is to extract ONLY the positive or abnormal findings from a clinical vignette and write them as a single concise paragraph of 3-5 sentences.
Do NOT include normal or negative examination findings (e.g., findings of 0/0, absent, or within normal limits).
Include ALL positive or abnormal findings present — do not omit any finding that could indicate a neuropathic, headache, or musculoskeletal disorder.
Focus on: pain quality, location, distribution, triggers, duration, temporal pattern, associated symptoms, and any positive sensory or neurological findings.
This summary will be used as a search query against clinical guideline databases.'''

    user_prompt = f'''Clinical vignette:

{vignette_text}

Write the positive-findings summary now.'''

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        max_completion_tokens=300,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    summary = response.choices[0].message.content.strip()
    print(f"\nPRE-RETRIEVAL QUERY SUMMARY:\n{summary}\n")
    return summary

def retrieve_mixed_chunks(query_text: str, collection):
    query_embedding = embed_query(query_text)
    mixed_documents = []
    mixed_metadata = []
    fallback_triggered = False

    for guideline in ["dc", "ichd", "icop"]:
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where={"corpus": {"$eq": guideline}},
                include=["documents", "distances", "metadatas"]
            )
            if results["documents"] and len(results["documents"][0]) > 0:
                mixed_documents.extend(results["documents"][0])
                mixed_metadata.extend(results["metadatas"][0])
                print(f"  [OK] Retrieved {len(results['documents'][0])} chunks from [{guideline}]")
            else:
                print(f"  [WARNING] No chunks retrieved from [{guideline}] — check folder name or corpus content.")
        except Exception as e:
            print(f"  [ERROR] Could not retrieve from [{guideline}]: {e}")

    if not mixed_documents:
        raise RuntimeError(
            "[RETRIEVAL FAILED] No chunks retrieved from any source. "
            "Check corpus folders are named exactly: dc, ichd, icop "
            "and that markdown files exist in each folder."
        )

    return mixed_documents, mixed_metadata, fallback_triggered

def build_guideline_excerpt(retrieved_chunks):
    return "\n\n".join(chunk.strip() for chunk in retrieved_chunks)

def run_arch3_initial(vignette_text: str, retrieved_chunks) -> str:
    guideline_excerpt = build_guideline_excerpt(retrieved_chunks)

    system_prompt = '''You are a dentist specializing in orofacial pain and temporomandibular disorders (TMD).

You will be provided with a patient vignette AND retrieved excerpts from three clinical guidelines: DC/TMD, ICOP, and ICHD-3.

Use only the clinical findings from the vignette for your assessment. Do not add or assume details that are not stated in the vignette.

Use the full term first followed by the abbreviation in brackets (e.g., temporomandibular joint (TMJ)), then use the abbreviation.

The provided case vignette contains muscle and temporomandibular joint (TMJ) palpation scores recorded separately for the right and left sides. Use these scores to assess the presence and severity of muscle and joint pain:
- 0 = no pain
- 1 = tenderness
- 2 = pain
- 3 = pain with referral
Think step-by-step like a clinician:

1. Identify all vignette findings: symptoms, examination (R/L), history.

2. Use the most relevant expert guideline framework to support your reasoning. Use guideline-based reasoning to improve diagnostic accuracy, but do not force unnecessary classification detail if the vignette supports a simpler clinical diagnosis.

3. Determine the primary diagnosis or diagnoses that best explain the vignette findings.
Use one diagnosis when a single diagnosis is sufficient.
Use more than one diagnosis only when the vignette clearly supports co-existing conditions that are each independently necessary to explain the case.
Use a clear standard diagnostic label.
Use the most specific diagnosis supported by the vignette findings.
If the vignette does not support a specific subtype, use the broader parent diagnosis instead.
Avoid listing overlapping labels that describe the same disorder at different levels of specificity.

4. Rank your top 5 differential diagnoses from most likely to least likely.

Write the final output using EXACTLY this structure:
List only the main diagnosis and diagnoses needed to explain the case.
Do not list multiple labels that refer to the same disorder at different levels of specificity.
Use standard diagnostic labels in the Primary diagnosis section, not descriptive sentence-style diagnoses.
Put explanatory details such as location, laterality, referral pattern, or descriptive qualifiers in Clinical reasoning unless they are part of the formal diagnosis name.

1) Primary diagnosis
   - [Formal diagnostic labels only]

2) Differential diagnoses
   - Give up to 5 differential diagnoses ranked in descending order from most likely to least likely

3) Clinical reasoning
   - Explain reasoning for each primary diagnosis
   - Provide reasoning for each differential diagnosis and why it is ranked where it is

4) Next steps
   List 3 to 5 of the most important next steps only. Each step must 
   directly confirm or refute the primary diagnosis or ensure patient safety.
   Always prioritize conservative approaches before irreversible interventions.
   - Specific clinical tests or examinations needed to confirm or refute diagnosis
   - Imaging if indicated
   - Referral if indicated

If the retrieved guideline excerpts do not contain sufficient diagnostic criteria to support a specific diagnosis, explicitly state in the clinical reasoning section: 'The retrieved corpus did not contain sufficient guideline content to confirm this diagnosis; reasoning is based on vignette findings alone.'

Keep all reasoning grounded in the vignette findings and the retrieved guideline excerpts.'''

    user_prompt = f'''<GUIDELINE_EXCERPTS>
{guideline_excerpt}
</GUIDELINE_EXCERPTS>

Clinical vignette:

{vignette_text}

Using the vignette text and the retrieved guideline excerpts, give me your diagnosis and next steps as per the instructions given to you.'''

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        max_completion_tokens=2000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

def run_arch3_refine(vignette_text: str, retrieved_chunks, initial_output: str) -> str:
    guideline_excerpt = build_guideline_excerpt(retrieved_chunks)

    system_prompt = '''You are reviewing a structured diagnostic assessment for an orofacial pain case.

Act as a careful clinical reviewer, not a second diagnostician.

Check whether the primary diagnosis, differential diagnoses and their ranking, clinical reasoning, and next steps are correct, supported by the vignette, consistent, and clearly written.

If the assessment is already good enough, keep it as it is.

Only change it if there is a clear problem, such as:
- a wrong diagnosis
- reasoning that is not supported by the vignette
- inconsistency
- an incorrect differential ranking
- missing important reasoning
- unsafe or unnecessary next steps
- unclear wording

If changes are needed, make only the smallest necessary corrections.
Do not replace the primary diagnosis, re-rank the differentials, or add new details unless this clearly improves diagnostic accuracy, safety, or clarity.
Do not add facts that are not stated in or clearly supported by the vignette.
Do not use overlapping labels for the same disorder.
Check whether the primary diagnosis section includes only the diagnosis or diagnoses that are independently supported and necessary to explain the vignette.
Use standard diagnostic labels in the Primary diagnosis section, not descriptive sentence-style diagnoses.
Use the most specific diagnosis supported by the vignette findings.
If the vignette does not support a specific subtype, revise to the broader supported diagnosis instead.
If explanatory wording such as location, referral pattern, or descriptive qualifiers is not part of the formal diagnosis name, move it to Clinical reasoning rather than leaving it in the diagnosis label.

ADDITIONAL CHECKS FOR RETRIEVED GUIDELINE USE:
- Corpus Gap: If the initial output cited guideline content that is NOT present in the retrieved excerpts, flag or remove that claim. Only guideline content explicitly present in the provided excerpts may be cited. If no relevant guideline content was retrieved for the primary diagnosis, the reasoning must state: 'The retrieved corpus did not contain sufficient guideline content to confirm this diagnosis; reasoning is based on vignette findings alone.'

Keep the same output structure and headings exactly:
1) Primary diagnosis
2) Differential diagnoses
3) Clinical reasoning
4) Next steps

Return the final version only.'''

    user_prompt = f'''<GUIDELINE_EXCERPTS>
{guideline_excerpt}
</GUIDELINE_EXCERPTS>

Clinical vignette:

{vignette_text}

Initial diagnostic assessment:

{initial_output}

Please review this assessment, reflect on whether it is accurate, safe, and well supported by both the vignette and the retrieved guideline excerpts, and then provide the final version in the same structure.'''

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        max_completion_tokens=2000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    collection = reset_and_store_corpus() if REBUILD_CORPUS else get_or_build_corpus()
    vignette_text = load_text(vignette_path)

    retrieval_query = summarize_vignette_for_retrieval(vignette_text)

    print("\nRETRIEVED SOURCES\n")

    retrieved_chunks, retrieved_metadata, fallback_triggered = retrieve_mixed_chunks(
        retrieval_query,
        collection
    )

    for idx, meta in enumerate(retrieved_metadata, 1):
        print(
            f"  {idx}. source={meta['source']}  corpus={meta.get('corpus')}  "
            f"chunk_index={meta['chunk_index']}  length={meta['length']}"
        )

    initial_output = run_arch3_initial(vignette_text, retrieved_chunks)
    final_output = run_arch3_refine(vignette_text, retrieved_chunks, initial_output)

    print("\nARCHITECTURE 3 OUTPUT\n")
    print(f"Case ID: {vignette_path.stem.upper()}")
    print(f"Model: {model_name}")
    print("\nINITIAL OUTPUT\n")
    print(initial_output)
    print("\nFINAL OUTPUT\n")
    print(final_output)

    output_dir.mkdir(parents=True, exist_ok=True)

    case_id = vignette_path.stem.upper()
    architecture = "arch3"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_filename = output_dir / f"{case_id}_{architecture}_{timestamp}.json"
    md_filename = output_dir / f"{case_id}_{architecture}_{timestamp}v10.md"

    json_output = {
        "case_id": case_id,
        "architecture": architecture,
        "model": model_name,
        "timestamp": timestamp,
        "vignette_file": str(vignette_path),
        "corpus_dir": str(corpus_dir),
        "retrieval_fallback_triggered": fallback_triggered,
        "retrieval_query_used": retrieval_query,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_metadata": retrieved_metadata,
        "initial_output": initial_output,
        "final_output": final_output,
    }

    with open(json_filename, "w", encoding="utf-8") as jf:
        json.dump(json_output, jf, indent=4)

    md_content = f"""# Architecture 3 Output

- **Case ID:** {case_id}
- **Architecture:** {architecture}
- **Model:** {model_name}
- **Timestamp:** {timestamp}
- **Vignette file:** {vignette_path}
- **Corpus directory:** {corpus_dir}
- **Retrieval fallback triggered:** {fallback_triggered}
- **Retrieval query used (pre-summarized):** {retrieval_query}

---

## Initial Output

{initial_output}

---

## Final Output

{final_output}
"""

    with open(md_filename, "w", encoding="utf-8") as mf:
        mf.write(md_content)

    print(f"\nSaved JSON to: {json_filename}")
    print(f"Saved MD to: {md_filename}")

    if fallback_triggered:
        print("\n  WARNING: Fallback was triggered — naive pooled retrieval was used.")
        print("Check corpus folder names (dc, ichd, icop) and re-run.")
    else:
        print("\n Source-constrained retrieval completed. No fallback triggered.")