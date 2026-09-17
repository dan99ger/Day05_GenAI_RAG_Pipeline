# ⚡ Day 05: Generative AI & Retrieval-Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-Integration-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready Retrieval-Augmented Generation (RAG) pipeline designed to chunk, embed, index, and retrieve precise information from private context documents using LangChain and local vector stores.

---

## 🛠️ Key Features

* **Text Chunking:** Recursive document splitting with overlapping windows to preserve semantic continuity.
* **Vector Embeddings:** Local embedding generation using Hugging Face's `all-MiniLM-L6-v2`.
* **Persistent Indexing:** Vector indexing and similarity search powered by **ChromaDB**.
* **Contextual Prompting:** Automatic retrieval and context extraction for accurate, LLM-ready prompt engineering.

---

## 📂 Repository Structure

```text
Day05_GenAI_RAG_Pipeline/
├── company_policy.txt    # Context document / Knowledge base
├── rag_pipeline.py       # Core RAG chunking, embedding & retrieval engine
├── .gitignore
└── README.md

🚀 How to Run
1. Setup Environment
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install langchain langchain-community langchain-chroma langchain-huggingface chromadb sentence-transformers pypdf python-dotenv
2. Execute RAG Pipeline
Bash
python rag_pipeline.py
Part of the 7-Day Machine Learning Engineering Challenge.
