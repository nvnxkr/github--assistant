# 🔍 GitHub Assistant — AI-Powered Codebase Navigator

> **AI assistant for understanding and analyzing GitHub repositories using RAG + FAISS vector store.**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-orange)](https://faiss.ai)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://langchain.com)

---

## Overview

**GitHub RAG Assistant** is an intelligent system that helps developers understand and analyze GitHub repositories. Instead of manually navigating complex codebases, you can ask questions in natural language and get accurate answers with source citations.

### How It Works

1. **Ingest** — Load GitHub repositories or ZIP files
2. **Index** — Generate semantic embeddings and store in FAISS vector database  
3. **Query** — Ask questions about the code using natural language
4. **Retrieve** — Multi-agent system finds relevant code and provides answers with citations

### Key Concepts

- **RAG (Retrieval Augmented Generation)** — Retrieves relevant code context before generating answers
- **FAISS Vector Store** — Fast semantic search for finding similar code chunks
- **Multi-Agent System** — Routes different types of questions to specialized agents:
  - Code Retrieval Agent (code Q&A)
  - Architecture Agent (system design)
  - Security Agent (vulnerability detection)
  - Bug Analysis Agent (bug detection)
- **LangChain** — Framework for building LLM-powered applications
- **Embeddings** — Uses BAAI/bge-large-en-v1.5 for semantic code understanding

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| **Frontend** | Streamlit | 1.38.0 |
| **Backend** | Flask | 3.0.3 |
| **Vector Store** | FAISS | 1.8.0 |
| **Embeddings** | Sentence Transformers | 3.0.1 |
| **LLM Framework** | LangChain | 0.2.16 |
| **Multi-Agent** | LangChain prompt chains | 0.2.16 |
| **LLM Providers** | Mistral AI, Groq | 1.0.3, 0.1.6 |
| **Repository Tools** | GitPython, PyGithub | 3.1.43, 2.3.0 |
| **Database** | SQLAlchemy + SQLite | 2.0.32 |
| **Server** | Gunicorn | 22.0.0 |
| **Python** | Python | 3.12+ |

---

## Installation

### Prerequisites

- Python 3.12 or higher
- Git
- pip (Python package manager)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/github-rag-assistant
cd github-rag-assistant

# 2. Create a virtual environment
uv venv .venv

# 3. Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate

# On Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Configure environment variables
cp .env.example .env
# Edit .env and set your API keys:
# - MISTRAL_API_KEY=sk-your-key (for Mistral)
# - GROQ_API_KEY=gsk-your-key (for Groq)
# - LLM_PROVIDER=mistral or groq
```

### Running the Application

```bash
# Start the Streamlit UI
streamlit run main.py

# The browser opens automatically. If it does not, visit:
# http://localhost:8501
```

The legacy Flask API is still available with `python server.py` for programmatic
clients.

The Streamlit sidebar provides an **LLM provider** selector for `Mistral` or
`Groq`. Configure both keys in `.env` if you want to switch between them:

```env
MISTRAL_API_KEY=your-mistral-key
GROQ_API_KEY=your-groq-key
GROQ_MODEL=llama-3.3-70b-versatile
```

### Troubleshooting LLM rate limits

If the UI reports a `429 rate_limited` response from Mistral, repository
indexing has completed but the chat provider rejected the answer request.
Wait for the provider limit to reset, verify `MISTRAL_API_KEY` and account
quota, or select another configured provider in `.env`:

```env
LLM_PROVIDER=groq
# or
LLM_PROVIDER=local
```

---

## Quick Example

1. **Upload a Repository** — Provide a GitHub URL or upload a ZIP file
2. **Wait for Indexing** — The system will analyze and index the codebase
3. **Ask Questions** — Query the code:
   - "What is the main function of this project?"
   - "How does authentication work?"
   - "Find potential security issues"
   - "Explain the system architecture"

---

## Environment Configuration

Create a `.env` file in the project root:

```env
LLM_PROVIDER=mistral          # mistral | groq | local
MISTRAL_API_KEY=...
GROQ_API_KEY=...
EMBEDDING_PROVIDER=sentence_transformers
SENTENCE_TRANSFORMER_MODEL=BAAI/bge-large-en-v1.5
GITHUB_TOKEN=                  # optional, for private repos
TOP_K=5
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
```

---

## License

MIT License - See LICENSE file for details

---

**Built with ❤️ for developers**
