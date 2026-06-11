import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Generate a response to a prompt
template = PromptTemplate(
input_variables=["topic"],
template="Write a short paragraph about {topic}."
)
response = llm.invoke(template.format(topic="RAG"))
print(response.content)