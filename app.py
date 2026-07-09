import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from utils import (
    PatchContextResponse,
    format_context,
    extract_citations,
    clean_answer,
    compute_verification_score
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_config():
    config_path = os.path.join(BASE_DIR, "config.pkl")
    if not os.path.exists(config_path):
        st.error("config.pkl not found. Run the notebook first.")
        st.stop()
    with open(config_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_vector_store(_config):
    embeddings = HuggingFaceEmbeddings(
        model_name=_config["EMBEDDING_MODEL"],
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    expected_name = _config["FAISS_INDEX_PATH"]
    index_path = os.path.join(BASE_DIR, expected_name)

    if not os.path.isdir(index_path) or not os.path.exists(os.path.join(index_path, "index.faiss")):
        found = False
        for item in os.listdir(BASE_DIR):
            item_path = os.path.join(BASE_DIR, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "index.faiss")):
                index_path = item_path
                found = True
                break

        if not found:
            st.error("FAISS index not found.")
            st.info(f"Expected: {expected_name}")
            st.stop()

    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_reranker(_config):
    return CrossEncoder(_config["RERANKER_MODEL"])


@st.cache_resource
def load_llm(api_key):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0
    )


def answer_question(question, vector_store, reranker, llm):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7}
    )

    retrieved_docs = retriever.invoke(question)
    pairs = [[question, doc.page_content] for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in scored[:5]]

    context = format_context(reranked_docs)

    prompt = f"""You are a FastAPI repository historian. Answer using ONLY the context below.

Context:
{context[:2000]}

Question: {question}

Rules:
- Answer directly. Never say "Yes, the question is answerable" or similar meta-phrases.
- If the context lacks evidence, state exactly: "No direct evidence found in the retrieved commits/PRs/issues."
- Cite specific PR/Issue numbers when available.
- Keep the answer factual and concise.

Answer:"""

    raw_answer = llm.invoke(prompt).content.strip()
    answer = clean_answer(raw_answer)

    verification_score = compute_verification_score(answer, context)
    is_verified = verification_score >= 0.3
    citations = extract_citations(reranked_docs)

    warning = ""
    if not is_verified:
        warning = "Low overlap with retrieved context. Verify citations carefully."
    if "no direct evidence" in answer.lower():
        warning = "Limited evidence found in retrieved context."
    if not any(citations.values()):
        warning = "No specific citations found."

    return PatchContextResponse(
        answer=answer,
        citations=citations,
        is_verified=is_verified,
        verification_score=verification_score,
        warning=warning
    )


st.set_page_config(page_title="PatchContext", page_icon="🔍", layout="wide")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to your .env file.")
    st.stop()

with st.spinner("Loading models and index..."):
    config = load_config()
    vector_store = load_vector_store(config)
    reranker = load_reranker(config)
    llm = load_llm(GROQ_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

with st.sidebar:
    st.header("🔍 About PatchContext")
    st.markdown(f"""
This RAG pipeline searches through the **{config['REPO_FULL_NAME']}** repository:
- Commit history
- Pull requests
- Issue discussions

All answers are grounded in actual developer conversations with clickable citations.
""")

    st.divider()

    st.header("💡 Sample Questions")
    samples = [
        "Why does FastAPI use Pydantic for validation?",
        "Why was dependency injection added?",
        "How does FastAPI achieve high performance?",
        "Why does FastAPI generate automatic documentation?",
        "What is the purpose of response_model?"
    ]

    for sample in samples:
        if st.button(sample, key=f"btn_{sample}", use_container_width=True):
            st.session_state.pending_question = sample

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

st.title("🔍 PatchContext")
st.markdown("*Ask questions about FastAPI design decisions, features, and implementation history*")
st.success("✓ All models loaded and ready")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about FastAPI...")

if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching repository history..."):
            response = answer_question(
                user_input,
                vector_store,
                reranker,
                llm
            )
            formatted = response.format_response()
            st.markdown(formatted)

    st.session_state.messages.append({
        "role": "assistant",
        "content": formatted
    })

    st.rerun()