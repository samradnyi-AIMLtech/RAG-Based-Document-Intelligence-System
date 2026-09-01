import os
import re
import tempfile

import faiss
import numpy as np
import streamlit as st
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Document Intelligence System")
st.caption("Multi-PDF • Ask • Summarize • Compare • Extract • Research")


# -----------------------------
# Cached models
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading language model...")
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# -----------------------------
# PDF / text helpers
# -----------------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append(
                {
                    "document": os.path.basename(pdf_path),
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(pages, chunk_size=800, overlap=150):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    result = []
    step = chunk_size - overlap

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                result.append(
                    {
                        "text": chunk_text,
                        "page": page["page"],
                        "document": page["document"],
                    }
                )

            start += step

    return result


def build_index(chunks, embedding_model):
    if not chunks:
        raise ValueError("No extractable text was found in the uploaded PDFs.")

    texts = [chunk["text"] for chunk in chunks if chunk["text"].strip()]

    if not texts:
        raise ValueError("The PDFs contain no extractable text.")

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve_documents(question, chunks, index, embedding_model,
                        selected_documents=None, top_k=5):
    if not chunks or index is None:
        return []

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    # Search extra candidates because document filtering happens after FAISS search.
    search_k = min(max(top_k * 10, top_k), index.ntotal)

    distances, indices = index.search(question_embedding, search_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        if (
            selected_documents
            and chunk["document"] not in selected_documents
        ):
            continue

        results.append(
            {
                "text": chunk["text"],
                "page": chunk["page"],
                "document": chunk["document"],
                "distance": float(distance),
            }
        )

        if len(results) >= top_k:
            break

    return results


# -----------------------------
# Prompt / generation
# -----------------------------
def create_prompt(question, retrieved_documents, mode="Ask"):
    context_parts = []

    for i, doc in enumerate(retrieved_documents, start=1):
        context_parts.append(
            f"""SOURCE {i}
DOCUMENT: {doc["document"]}
PAGE: {doc["page"]}
CONTENT:
{doc["text"]}
-------------------------"""
        )

    context = "\n\n".join(context_parts)

    instructions = {
        "Ask": "Answer the user's question directly.",
        "Summarize": (
            "Summarize the relevant information from the uploaded documents. "
            "Use clear bullet points."
        ),
        "Compare": (
            "Compare the information from the provided documents. "
            "Clearly identify similarities, differences, and important findings."
        ),
        "Extract": (
            "Extract the most important factual information from the documents. "
            "Use structured bullet points."
        ),
        "Research": (
            "Give a detailed research-style answer using only the uploaded "
            "documents. Organize the answer clearly."
        ),
    }

    instruction = instructions.get(mode, instructions["Ask"])

    return f"""You are a RAG document intelligence assistant.

IMPORTANT RULES:
1. Use ONLY the provided context.
2. Do not invent information.
3. If the answer is not available, say:
"I could not find the answer in the uploaded documents."
4. Mention document names and page numbers when relevant.

TASK:
{instruction}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def generate_answer(prompt, tokenizer, llm):
    # FLAN-T5 has a finite input context. Truncation keeps the app from
    # failing when several retrieved chunks are large.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# -----------------------------
# Session state
# -----------------------------
if "all_pages" not in st.session_state:
    st.session_state.all_pages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Document Setup")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_clicked = st.button(
        "⚙️ Process PDFs",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    mode = st.radio(
        "Choose Intelligence Mode",
        ["Ask", "Summarize", "Compare", "Extract", "Research"],
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=8,
        value=5,
    )


# -----------------------------
# Process PDFs
# -----------------------------
if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        pages = []

        progress = st.progress(0, text="Reading PDFs...")

        for i, uploaded_file in enumerate(uploaded_files):
            suffix = ".pdf"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                pdf_pages = extract_text_from_pdf(tmp_path)

                # Preserve the uploaded filename rather than the temporary filename.
                for page in pdf_pages:
                    page["document"] = uploaded_file.name
                    page["text"] = clean_text(page["text"])

                pages.extend(pdf_pages)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"Reading {uploaded_file.name}...",
            )

        chunks = create_chunks(pages)

        try:
            embedding_model = load_embedding_model()
            index = build_index(chunks, embedding_model)
        except Exception as exc:
            st.session_state.all_pages = []
            st.session_state.chunks = []
            st.session_state.index = None
            st.error(f"Could not process the PDFs: {exc}")
        else:
            st.session_state.all_pages = pages
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.processed_documents = sorted(
                {chunk["document"] for chunk in chunks}
            )

            progress.empty()
            st.success(
                f"Processed {len(st.session_state.processed_documents)} "
                f"document(s), {len(pages)} page(s), and {len(chunks)} chunk(s)."
            )


# -----------------------------
# Document selector
# -----------------------------
documents = st.session_state.processed_documents

if documents:
    selected_documents = st.multiselect(
        "📄 Select PDFs to search",
        options=documents,
        default=documents,
    )
else:
    selected_documents = []

st.divider()


# -----------------------------
# Main question area
# -----------------------------
question = st.text_area(
    "Ask your documents",
    placeholder="Example: What are the main technologies discussed?",
    height=120,
)

ask_clicked = st.button(
    "🔎 Run RAG",
    type="primary",
)


if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    elif st.session_state.index is None:
        st.warning("Please upload and process your PDFs first.")
    elif not selected_documents:
        st.warning("Please select at least one PDF.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            embedding_model = load_embedding_model()

            retrieved = retrieve_documents(
                question=question,
                chunks=st.session_state.chunks,
                index=st.session_state.index,
                embedding_model=embedding_model,
                selected_documents=selected_documents,
                top_k=top_k,
            )

            if not retrieved:
                answer_text = (
                    "I could not find relevant information in the selected documents."
                )
            else:
                prompt = create_prompt(
                    question,
                    retrieved,
                    mode,
                )

                tokenizer, llm = load_llm()
                answer_text = generate_answer(
                    prompt,
                    tokenizer,
                    llm,
                )

        st.subheader("💡 Answer")
        st.markdown(answer_text)

        if retrieved:
            st.subheader("📚 Sources")

            for source in retrieved:
                with st.expander(
                    f"📄 {source['document']} — Page {source['page']}"
                ):
                    st.write(source["text"])
                    st.caption(
                        f"FAISS distance: {source['distance']:.4f}"
                    )


# -----------------------------
# Footer
# -----------------------------
st.caption(
    "Note: This demo processes uploaded PDFs in the running app session. "
    "It does not permanently store uploaded documents."
)
