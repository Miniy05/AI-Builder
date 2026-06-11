from openai import OpenAI

client = OpenAI(api_key="")

user_input = input("You: ")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.7,
    messages=[
        {"role": "user", "content": user_input}
    ]
)