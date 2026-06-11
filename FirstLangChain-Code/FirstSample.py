from openai import OpenAI

client = OpenAI(api_key="")


prompt = """
You are a Python teacher.
Explain loops in Hindi with examples.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"user","content":prompt}
    ]
)

print(response.choices[0].message.content)
    