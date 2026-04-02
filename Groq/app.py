import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

if "vectors" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:50]
    )
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, st.session_state.embeddings
    )

st.title("ChatGroq Demo")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

qa_prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Question: {input}
"""
)

document_chain = qa_prompt | llm | StrOutputParser()
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | document_chain
)

user_query = st.text_input("Input your prompt here")

if user_query:
    start = time.process_time()

    # Final answer (string)
    response = retrieval_chain.invoke(user_query)

    # Retrieved docs (list[Document]) for expander display
    context_docs = retriever.invoke(user_query)

    print("Response time :", time.process_time() - start)
    st.write(response)

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(context_docs):
            st.write(doc.page_content)
            st.write("-------------------------")