# RAG-Based-Document-Intelligence-System
RAG-Based Document Intelligence System | AI-Powered PDF Question Answering
### Description

A RAG (Retrieval Augmented Generation) architecture that is smart enough to mine, retrieve, and generate insights from a large number of documents. The architecture consists of document processing, semantic search, vector databases, and LLMs for generating accurate answers to any query posed to it.

### Key Features

* **Multi-PDF** — Upload and analyze multiple PDF files at once.
* **Semantic Search** — Uses `all-MiniLM-L6-v2` to create document embeddings.
* **FAISS Vector Search** — Efficiently retrieves the most relevant document information.
* **AI Answers** — Generates answers using Google’s `FLAN-T5-Base` model.
* **Multiple Intelligence Modes** — Ask, Summarize, Compare, Extract, and Research.
* **Ask** — Answers questions based on the uploaded documents.
* **Summarize** — Provides concise summaries of document content.
* **Compare** — Identifies similarities, differences, and key insights across documents.
* **Extract** — Extracts important factual information from documents.
* **Research** — Generates research-style answers using retrieved document context.
* **Document Selection** — Allows users to select specific documents for information retrieval.
* **Source References** — Displays the document name and page number for retrieved information.
* **Interactive User Interface** — Built with Gradio for easy document analysis.
* **Context-Based Responses** — Provides answers using only retrieved document context to avoid unsupported information.

### System Workflow

The system follows a standard RAG pipeline:

PDF Upload → Text Extraction → Text Cleaning → Chunking → Embeddings → FAISS Index → Semantic Retrieval → Prompt Construction → FLAN-T5 Generation → Answer + Sources

### 🛠️Tech Stack
* **Python**
* **PyPDF**
* **Sentence Transformers**
* **FAISS**
* **Hugging Face Transformers**
* **FLAN-T5**
* **PyTorch**
* **Gradio**
* **NumPy**


### 📌Use Cases

This project can be used for:

* **Research Paper Analysis**
* **Technical Documentation Search**
* **Academic Document Analysis**
* **Knowledge-Base Question Answering**
* **Multi-Document Comparison**
* **Information Extraction from PDFs**
* **Research Assistance**


RAG Document Intelligence System is an AI-powered document analysis application that allows users to upload and interact with multiple PDF files. It extracts and chunks document text, generates semantic embeddings using Sentence Transformers, and uses FAISS for similarity-based retrieval. A FLAN-T5 model then generates context-aware responses.





