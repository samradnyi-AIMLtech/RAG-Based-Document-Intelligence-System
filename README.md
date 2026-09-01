# RAG Document Intelligence System

A Streamlit version of the RAG Document Intelligence project.

## Features

- Upload multiple PDFs
- Extract and clean PDF text
- Chunk documents
- Generate embeddings with `all-MiniLM-L6-v2`
- Search with FAISS
- Generate answers with `google/flan-t5-base`
- Ask / Summarize / Compare / Extract / Research modes
- Show source documents and page numbers

## Deploy

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Open Streamlit Community Cloud.
4. Connect your GitHub account.
5. Select the repository and `app.py`.
6. Deploy.

The app will receive a public `streamlit.app` URL.

## Important

The app uses CPU-based models, so the first model load can take some time.
