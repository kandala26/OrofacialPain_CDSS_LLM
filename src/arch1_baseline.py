import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import json
from datetime import datetime

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found.")

client = OpenAI(api_key=api_key)

model_name = "gpt-5.4"
vignette_path = Path("data/vignettes/v6.md")
case_id = vignette_path.stem.upper()


def load_vignette(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def run_baseline(vignette_text: str) -> str:
    """Architecture 1: Baseline LLM with simple structured prompt."""

    SYSTEM_PROMPT = '''You are a dentist specializing in orofacial pain and temporomandibular disorders.

The provided case vignette contains muscle and temporomandibular joint (TMJ) palpation scores recorded separately for the right and left sides. Use these scores to assess the presence and severity of muscle and joint pain:
- 0 = no pain
- 1 = tenderness
- 2 = pain
- 3 = pain with referral. 
ALWAYS use this exact output format:

1) Primary diagnosis
   - Give the primary diagnosis based on the vignette.
   - If the vignette clearly supports more than one independent co-existing 
     condition, list all of them. Do not repeat the same thing or synonyms 
   - Use a standard clinical diagnostic label only.
    
2) Differential diagnoses
   - Give up to 5 differential diagnoses ranked in descending order from most likely to least likely

3) Clinical reasoning
   - Explain reasoning for primary diagnosis/ diagnoses
   - Provide reasoning for each differential diagnosis and why it is ranked where it is

4) Next steps
   - Suggest tests, exams, or next clinical steps. 
   
   Use the full term first followed by the abbreviation in brackets (e.g., temporomandibular joint (TMJ)), then use the abbreviation.

Use this format exactly. No other structure.'''

    USER_PROMPT = f'''Clinical vignette:

{vignette_text}

Using only the clinical information provided in the vignette above, apply your diagnostic knowledge to interpret the findings and provide the structured assessment as instructed. Do not introduce symptoms, history, imaging, or examination findings that are not stated in the vignette'''
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        max_completion_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
    )

    return response.choices[0].message.content




if __name__ == "__main__":
    vignette = load_vignette(vignette_path)
    result = run_baseline(vignette)

    print("\nMODEL OUTPUT\n")
    print(result)

    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    architecture = "arch1"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_filename = output_dir / f"{case_id}_{architecture}_{timestamp}.json"

    json_output = {
        "case_id": case_id,
        "architecture": architecture,
        "model": model_name,
        "timestamp": timestamp,
        "vignette_file": str(vignette_path),
        "response": result
    }

    with open(json_filename, "w", encoding="utf-8") as jf:
        json.dump(json_output, jf, indent=4)

    print(f"\nSaved JSON to: {json_filename}")

    md_filename = output_dir / f"{case_id}_{architecture}_{timestamp}v10.md"

    md_content = f"""# Model Output

- **Case ID:** {case_id}
- **Architecture:** {architecture}
- **Model:** {model_name}
- **Timestamp:** {timestamp}
- **Vignette file:** {vignette_path}

---

## Response

{result}
"""

    with open(md_filename, "w", encoding="utf-8") as mf:
        mf.write(md_content)

    print(f"\nSaved MD to: {md_filename}")