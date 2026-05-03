import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found.")

client = OpenAI(api_key=api_key)

model_name = "gpt-5.4"

BASE_DIR = Path(__file__).resolve().parent
vignette_path = BASE_DIR / "data" / "vignettes" / "v6.md"
output_dir = BASE_DIR / "data" / "outputs"


def load_vignette(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def run_arch2_initial(vignette_text: str) -> str:

    system_prompt = '''You are a dentist specializing in orofacial pain and temporomandibular disorders (TMD).

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

List only the main diagnosis or diagnoses needed to explain the case.
Do not list multiple labels that refer to the same disorder at different levels of specificity.
Use standard diagnostic labels in the Primary diagnosis section, not descriptive sentence-style diagnoses.

Write the final output using EXACTLY this structure:

1) Primary diagnosis
   - [Only primary diagnosis or diagnoses here]
    - List the primary diagnosis or diagnoses using formal diagnostic labels only.
   - If a single diagnosis best explains the vignette, list only that one.
   - If the vignette clearly and independently supports more than one co-existing 
     condition that cannot be explained by a single diagnosis, list all of them.
   - Do not list multiple diagnoses simply because they are related or overlapping.
   - Do not list negative findings or exclusions as a diagnosis. 
     Negative findings belong in Clinical reasoning only.

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

Keep all reasoning grounded in the vignette findings and the guidelines mentioned above.'''

    user_prompt = f'''Clinical vignette:

{vignette_text}

Using the vignette text and your expert knowledge, give me your diagnosis and next steps as per the instructions given to you.'''

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


def run_arch2_refine(vignette_text: str, initial_output: str) -> str:

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

Keep the same output structure and headings and format as initial output:
1) Primary diagnosis
2) Differential diagnoses
3) Clinical reasoning
4) Next steps

Return the final version only.'''

    user_prompt = f'''Clinical vignette:

{vignette_text}

Initial diagnostic assessment:

{initial_output}

Please review this assessment, reflect on whether it is accurate, safe, and well supported, and then provide the final version in the same structure.'''

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
    vignette_text = load_vignette(vignette_path)

    initial_output = run_arch2_initial(vignette_text)
    final_output = run_arch2_refine(vignette_text, initial_output)

    print("\nARCHITECTURE 2 OUTPUT\n")
    print(f"Case ID: {vignette_path.stem.upper()}")
    print(f"Model: {model_name}")

    print("\nINITIAL OUTPUT\n")
    print(initial_output)

    print("\nFINAL OUTPUT\n")
    print(final_output)

    output_dir.mkdir(parents=True, exist_ok=True)

    case_id = vignette_path.stem.upper()
    architecture = "arch2"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_filename = output_dir / f"{case_id}_{architecture}_{timestamp}.json"
    md_filename = output_dir / f"{case_id}_{architecture}_{timestamp}v10.md"

    json_output = {
        "case_id": case_id,
        "architecture": architecture,
        "model": model_name,
        "timestamp": timestamp,
        "vignette_file": str(vignette_path),
        "initial_output": initial_output,
        "final_output": final_output,
    }

    with open(json_filename, "w", encoding="utf-8") as jf:
        json.dump(json_output, jf, indent=4)

    md_content = f"""# Architecture 2 Output

- **Case ID:** {case_id}
- **Architecture:** {architecture}
- **Model:** {model_name}
- **Timestamp:** {timestamp}
- **Vignette file:** {vignette_path}

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