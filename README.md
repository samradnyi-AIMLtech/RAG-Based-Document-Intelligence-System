# RAG-Based-Document-Intelligence-System
RAG-Based Document Intelligence System | AI-Powered PDF Question Answering
### 1. Description

A RAG (Retrieval Augmented Generation) architecture that is smart enough to mine, retrieve, and generate insights from a large number of documents. The architecture consists of document processing, semantic search, vector databases, and LLMs for generating accurate answers to any query posed to it.

### Key Features

Multi-PDF — Allows you to upload and analyze multiple PDF files at once.
Semantic Search — Embeds chunks of your document using all-MiniLM-L6-v2.
FAISS Vector Search — Provides an efficient search of the most relevant document information.
AI Answers — Generates answers based on the retrieved context using Google's FLAN-T5-Base model.
Multiple Intelligence Modes
Ask — Answer questions related to your uploaded documents.
Summarize — Summarize the information in the documents.
Compare — Find common features, differences, and insights.
Extract — Extract key factual information from your documents.
Research — Provide research-style answers.
Document Selection — Choose which documents you want to use for the retrieval of information.
Source References — Shows the name of the document and page number where the information is retrieved from.
Interactive User Interface — Built using Gradio for easy-to-use document analysis.
Context-Based Responses — Context-only answers designed to not provide any unsupported information.

### System Workflow

The system follows a standard RAG pipeline:

PDF Upload → Text Extraction → Text Cleaning → Chunking → Embeddings → FAISS Index → Semantic Retrieval → Prompt Construction → FLAN-T5 Generation → Answer + Sources

### 🛠️Tech Stack
Python
PyPDF
Sentence Transformers
FAISS
Hugging Face Transformers
FLAN-T5
PyTorch
Gradio
NumPy

### 📌Use Cases

This project can be used for:

Research paper analysis
Technical documentation search
Academic document analysis
Knowledge-base question answering
Multi-document comparison
Information extraction from PDFs
Research assistance

RAG Document Intelligence System is an AI-powered document analysis application that allows users to upload and interact with multiple PDF files. It extracts and chunks document text, generates semantic embeddings using Sentence Transformers, and uses FAISS for similarity-based retrieval. A FLAN-T5 model then generates context-aware responses.





