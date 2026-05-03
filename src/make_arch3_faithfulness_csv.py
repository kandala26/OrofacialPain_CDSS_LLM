import pandas as pd
import json
from pathlib import Path

output_dir = Path("data/outputs")
rows = []

for f in output_dir.glob("*_arch3_*.json"):
    with open(f, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    rows.append({
        "case_id": data["case_id"],
        "question": data["retrieval_query_used"],
        "contexts": data["retrieved_chunks"],
        "answer": data["final_output"],
        "initial_output": data["initial_output"],
        "final_output": data["final_output"],
        "json_file": f.name
    })

df = pd.DataFrame(rows)
df.to_csv("data/arch3_faithfulness.csv", index=False)

print(f"Saved {len(df)} rows to data/arch3_faithfulness.csv")
print(df[["case_id", "json_file"]].head())

import pandas as pd; df=pd.read_csv('/Users/moukthikareddy/PycharmProjects/orofacialpain_llm_cdss/src/data/arch3_faithfulness.csv'); print(df[['case_id','json_file']].to_string(index=False))