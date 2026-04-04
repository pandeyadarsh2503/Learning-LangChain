# 🦜🔗 Learning LangChain

A personal learning repository where I explored **LangChain** — the Python framework for building applications powered by Large Language Models (LLMs). This is **not a production project** — it's a hands-on playground covering key LangChain concepts from simple chatbots all the way to agents, RAG pipelines, and HuggingFace embeddings.

---

## 📁 Repository Structure

```
Learning LangChain/
│
├── Chatbot/                  # First steps: basic Streamlit chatbot with LLMs
│   ├── app.py                # Chatbot using Google Gemini (via LangChain)
│   └── localama.py           # Chatbot using local Ollama (llama3.1:8b)
│
├── API's/                    # Serving LangChain chains as a REST API
│   ├── app.py                # FastAPI + LangServe server (essay & poem routes)
│   └── client.py             # Streamlit client that hits the API endpoints
│
├── Groq/                     # RAG chatbot powered by Groq (ultra-fast inference)
│   └── app.py                # Web-scrapes LangSmith docs → FAISS → Groq LLM
│
├── RAG/                      # Retrieval-Augmented Generation experiments
│   ├── simple_rag.ipynb      # RAG basics: TextLoader, WebLoader, PyPDFLoader
│   ├── retriever.ipynb       # Full RAG chain: FAISS + Ollama LLM + retriever
│   ├── speech.txt            # Sample text (MLK "I Have a Dream" speech)
│   ├── Medical_book.pdf      # Sample PDF (Gale Encyclopedia of Medicine)
│   ├── VoiceAgent.pdf        # Sample PDF (Voice Agent engineering assignment)
│   └── Heartbeat Script Design Doc.pdf  # Another sample PDF document
│
├── HuggingFace/              # HuggingFace embeddings + FAISS RAG pipeline
│   ├── huggingface.ipynb     # BAAI/bge-small-en-v1.5 embeddings + RAG chain
│   └── us_census/            # Sample PDFs (US Census health insurance reports)
│
└── MultiSearch Agent/        # LangChain agent with multiple search tools
    └── agents.ipynb          # Agent with Wikipedia + Arxiv + LangSmith retriever
```

---

## 📚 What I Learned — Module by Module

### 1. 🤖 Chatbot
**Concepts:** `ChatPromptTemplate`, `StrOutputParser`, LangSmith tracing, Streamlit UI

- **`app.py`** — A minimal Q&A chatbot using **Google Gemini** (`gemini-flash-latest`) through `langchain_google_genai`. Integrates **LangSmith tracing** (`LANGCHAIN_TRACING_V2`) to observe chain calls in real time.
- **`localama.py`** — Same chatbot but running **completely locally** using **Ollama** (`llama3.1:8b`). Demonstrates how to swap LLMs without changing the rest of the chain.

Key pattern learned: `prompt | llm | output_parser`

---

### 2. 🌐 API's (LangServe)
**Concepts:** FastAPI, `LangServe`, `add_routes`, serving chains as REST endpoints

- **`app.py`** — A **FastAPI** server that exposes two LangChain chains as HTTP endpoints:
  - `POST /essay/invoke` → writes a 100-word essay using Gemini
  - `POST /poem/invoke` → writes a 100-word poem using Gemini
- **`client.py`** — A **Streamlit** frontend that calls these endpoints via `requests.post(...)`, showing how to decouple the UI from the LLM backend.

Key pattern learned: wrapping any LangChain chain as a production-ready API in ~10 lines of code.

---

### 3. ⚡ Groq (RAG with Fast Inference)
**Concepts:** `WebBaseLoader`, `RecursiveCharacterTextSplitter`, FAISS, `ChatGroq`, RAG chain, Streamlit session state

- **`app.py`** — A full **Retrieval-Augmented Generation** Streamlit app that:
  1. Scrapes the LangSmith documentation website
  2. Splits it into chunks and stores embeddings in **FAISS** (using local Ollama `nomic-embed-text`)
  3. Answers user questions using **Groq** (`llama-3.1-8b-instant`) — one of the fastest inference APIs available
  4. Shows the retrieved source documents in an expandable panel

Key pattern learned: using Streamlit `session_state` to avoid re-loading embeddings on every UI interaction.

---

### 4. 📄 RAG (Retrieval-Augmented Generation)

**`simple_rag.ipynb`** — Explored multiple **document loaders** side by side:
| Loader | Source |
|---|---|
| `TextLoader` | Local `.txt` file (MLK speech) |
| `WebBaseLoader` | Wikipedia page (Artificial Intelligence) |
| `PyPDFLoader` | Local PDF (637-page Medical Book) |

Then compared two vector stores: **ChromaDB** vs **FAISS**, running similarity search on each.

---

**`retriever.ipynb`** — Built a full **RAG pipeline** from scratch:
1. Load `VoiceAgent.pdf` with `PyPDFLoader`
2. Embed with `OllamaEmbeddings` (`nomic-embed-text`) → store in **FAISS**
3. Query the vector store and pass results to a **context-aware prompt**
4. Run through a local **Ollama LLM** (`llama3.1:8b`)
5. Parse output with `StrOutputParser`

Also hit a real bug here — a `KeyError` from a prompt variable mismatch — good learning experience!

---

### 5. 🤗 HuggingFace
**Concepts:** `HuggingFaceBgeEmbeddings`, FAISS, `HuggingFacePipeline`, local RAG without any API keys

**`huggingface.ipynb`** — Learned how to run a full RAG pipeline **100% locally for free** using HuggingFace:
1. Load 18 US Census PDF reports → **316 document chunks**
2. Embed using **`BAAI/bge-small-en-v1.5`** (BGE = BAAI General Embedding) — a top-performing small embedding model
3. Store in **FAISS**
4. Answer questions using **`google/flan-t5-large`** via `HuggingFacePipeline`

Key pattern learned: the difference between embedding models (for retrieval) and generative models (for answering).

---

### 6. 🕵️ MultiSearch Agent
**Concepts:** LangChain Agents, Tool creation, `AgentExecutor`, multi-tool reasoning

**`agents.ipynb`** — Built a **multi-tool agent** that can reason over and call:
| Tool | What it does |
|---|---|
| `WikipediaQueryRun` | Answers general knowledge questions |
| `ArxivQueryRun` | Searches scientific papers |
| `create_retriever_tool` | Searches LangSmith docs (custom FAISS retriever) |

The agent uses **Google Gemini** as its reasoning backbone and the `hwchase17/openai-functions-agent` prompt from LangChain Hub. Successfully answered "Tell me about LangSmith" by combining Wikipedia + its own vector store. Also hit an Arxiv API rate-limit error (HTTP 429) — realistic learning moment!

---

## 🔑 Environment Setup

A template is provided — copy it and fill in your keys:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Learning-LangChain
```

| Key | Where to get it |
|---|---|
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `GROQ_API_KEY` | [Groq Console](https://console.groq.com/keys) |
| `LANGCHAIN_API_KEY` | [LangSmith](https://smith.langchain.com) → Settings → API Keys |

> **Note:** Never commit your `.env` file. It is already listed in `.gitignore`.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

All dependencies are listed in [`requirements.txt`](requirements.txt), grouped by purpose:

| Group | Key Packages |
|---|---|
| **Core LangChain** | `langchain`, `langchain-core`, `langchain-community` |
| **LLM Providers** | `langchain-google-genai`, `langchain-groq`, `langchain-ollama`, `langchain-huggingface` |
| **Vector Stores** | `faiss-cpu`, `chromadb` |
| **Embeddings / Models** | `sentence-transformers`, `transformers`, `torch` |
| **API Serving** | `langserve`, `fastapi`, `uvicorn` |
| **Document Loaders** | `pypdf`, `beautifulsoup4`, `wikipedia`, `arxiv` |
| **UI & Utilities** | `streamlit`, `python-dotenv`, `numpy` |

> **Tip:** If you only want to run a specific module, you don't need all packages — install just what that folder uses.

---

## 🚀 Running the Apps

### Chatbot (Gemini)
```bash
cd Chatbot
streamlit run app.py
```

### Chatbot (Local Ollama)
```bash
# Ensure Ollama is running: ollama serve
cd Chatbot
streamlit run localama.py
```

### API Server + Client
```bash
# Terminal 1 — start the server
cd "API's"
python app.py

# Terminal 2 — start the Streamlit client
cd "API's"
streamlit run client.py
```

### Groq RAG App
```bash
cd Groq
streamlit run app.py
```

### Jupyter Notebooks
```bash
# Open any folder in Jupyter
jupyter notebook
```

---

## 🛠️ Prerequisites

1. **Python 3.10+**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # then edit .env with your API keys
   ```
4. **Ollama** (for local LLM notebooks): [ollama.ai](https://ollama.ai)
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
5. **API Keys** needed: Google AI Studio, Groq Cloud, LangSmith (all free tiers available)

---

## 🗺️ Learning Journey

```
Chatbot (basics)
    ↓
LangServe APIs (serving chains)
    ↓
RAG: simple_rag (loaders & vector stores)
    ↓
RAG: retriever (full pipeline with local LLM)
    ↓
Groq (fast cloud inference + RAG)
    ↓
HuggingFace (100% free, local embeddings + LLM)
    ↓
Agents (multi-tool reasoning + AgentExecutor)
```

