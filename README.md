# RAG Document Intelligence System

Demo link - https://rag-doc-intelligence.streamlit.app/

A RAG-based Document Intelligence System that allows users to upload and interact with multiple PDF documents using AI-powered retrieval and analysis.The system supports multiple intelligence modes, including Ask, Summarize, Compare, Extract, and Research, enabling users to query individual or multiple documents and receive context-aware responses with relevant citations.
#### Tech Stack: Python • RAG • Sentence Transformers • FAISS • Qwen 2.5 • Gradio • Streamlit

## Key Features
- **Multi-PDF Support** – Upload and process multiple PDF documents simultaneously.
- **RAG-Based Q&A** – Retrieve relevant document context and generate accurate answers.
- **Multiple Intelligence Modes** – Ask questions, summarize documents, compare content, extract information, and perform research.
- **Rates & Values Extraction** – Identify important rates, values, percentages, amounts, and numerical information.
- **Semantic Embeddings** – Uses `all-MiniLM-L6-v2` from Sentence Transformers.
- **FAISS Vector Search** – Enables efficient similarity-based document retrieval.
- **Qwen LLM** – Uses `Qwen/Qwen2.5-3B-Instruct` for intelligent response generation.
- **Custom Prompting** – Uses a `create_prompt()` function to construct context-aware prompts.
- **Citations** – Provides document-grounded responses based on retrieved content.
- **Gradio Interface** – Interactive interface for testing the document intelligence pipeline.
- **Streamlit Deployment** – Web-based interface for uploading, processing, and querying documents.

   ### RAG Pipeline Overview
  PDF Upload → Text Processing → Embeddings → FAISS Retrieval → Context Retrieval → Prompt Creation → Qwen LLM → Document Intelligence → Answer + Citations.

## Project Summary
A RAG-based Document Intelligence System developed to gain hands-on experience in Generative AI, Large Language Models, and information retrieval. The project explores the complete RAG pipeline, including document processing, semantic embeddings, FAISS-based vector retrieval, prompt engineering, and LLM-powered response generation using Qwen 2.5.
The system supports multi-document Q&A, summarization, comparison, information extraction, research, and rates & values analysis, with interactive interfaces built using Gradio and Streamlit. The project focuses on practical experimentation with RAG workflows and building reliable, document-grounded AI applications.

