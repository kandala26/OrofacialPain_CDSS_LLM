import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

if not key:
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env file location and name.")

client = OpenAI(api_key=key)

resp = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Say 'API connected'."}],

)

print(resp.choices[0].message.content)