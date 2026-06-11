import streamlit as st
from openai import OpenAI

client = OpenAI(api_key="")


st.title("Mini AI Assistant")

user_input = st.chat_input("Type your message")

if user_input:

    st.chat_message("user").write(user_input)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"user","content":user_input}
        ]
    )

    ai_reply = response.choices[0].message.content

    st.chat_message("assistant").write(ai_reply)