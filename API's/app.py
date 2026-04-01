from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(
    title = "Langchain Server",
    version= "1.0",
    description="A simple API server"
)

add_routes(
    app,
    ChatGoogleGenerativeAI(model="gemini-flash-latest"),
    path="/gemini"
)

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")
llm = Ollama(model="llama3.1:8b")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words.")
prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} with 100 words.")

add_routes(
    app,
    prompt1|model,
    path="/essay"
)

add_routes(
    app,
    prompt2|model,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)