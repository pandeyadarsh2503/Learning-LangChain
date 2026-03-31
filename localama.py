from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
import dotenv

dotenv.load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
#LangSmith Tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

#Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a helpful assistant. Please respond to the user's query."),
        ("user","Question:{question}")
    ]
)

st.title("Langchain Demo  with OPENAI API")
input_text = st.text_input("Enter your question here:")

llm = Ollama(model="llama3.1:8b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
