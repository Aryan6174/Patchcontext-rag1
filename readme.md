# PatchContext

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://patchcontext-rag1-aryan-dadhich.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-0467DF)](https://github.com/facebookresearch/faiss)

PatchContext is a retrieval-augmented generation application that answers engineering questions about the development history of the [FastAPI](https://github.com/fastapi/fastapi) repository.

Instead of searching through GitHub manually, users can ask questions such as:

- Why was this feature added?
- Why was this design decision made?
- Which issue discusses this behavior?
- Which pull request changed this implementation?
- What repository evidence explains this feature?

PatchContext searches through commit messages, pull requests, and issues before generating an answer. It also displays clickable links to the original GitHub sources.

## Live Demo

Try the deployed application:

### [Open PatchContext](https://patchcontext-rag1-aryan-dadhich.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://patchcontext-rag1-aryan-dadhich.streamlit.app/)

The application may take a few seconds to start when it has been inactive because Streamlit Community Cloud puts unused applications to sleep.

## How It Works

```text
FastAPI GitHub repository
          ↓
Commits, pull requests and issues
          ↓
LangChain Documents
          ↓
Text chunking
          ↓
Local Hugging Face embeddings
          ↓
FAISS vector index
          ↓
MMR retrieval
          ↓
CrossEncoder reranking
          ↓
Groq LLM
          ↓
Answer with GitHub citations
```

Only the highest-ranked retrieved chunks are sent to the language model. The complete repository history is never included in the generation prompt.

## Features

- Collects FastAPI commit messages through the GitHub API
- Collects pull request titles and descriptions
- Collects issue titles and bodies
- Includes both recent and older repository discussions
- Converts GitHub records into LangChain documents
- Stores commit SHAs, PR numbers, issue numbers, and URLs as metadata
- Splits long documents into smaller overlapping chunks
- Generates embeddings locally in batches
- Stores embeddings in a persistent FAISS index
- Uses MMR retrieval for relevant and diverse results
- Uses a CrossEncoder to rerank retrieved documents
- Uses Groq for fast answer generation
- Returns clickable GitHub citations
- Displays a verification score and warnings for weakly grounded answers
- Includes an optional NLI guard in the notebook
- Provides a Streamlit chat interface
- Saves the FAISS index so embeddings do not need to be regenerated for every run

## Technology Stack

| Component | Technology |
|---|---|
| Repository | FastAPI |
| Data source | GitHub REST API |
| RAG framework | LangChain |
| Embeddings | `all-MiniLM-L6-v2` |
| Vector database | FAISS |
| Retrieval | Maximal Marginal Relevance |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer generation | Groq `llama-3.1-8b-instant` |
| Optional NLI model | `facebook/bart-large-mnli` |
| Web interface | Streamlit |

Embeddings and reranking run locally. Groq is used only for final answer generation.

## Repository

PatchContext currently uses the FastAPI repository:

- Repository: [fastapi/fastapi](https://github.com/fastapi/fastapi)
- Documentation: [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)

## Project Structure

```text
patchcontext/
├── app.py
├── utils.py
├── requirements.txt
├── config.pkl
├── README.md
├── .gitignore
├── PatchContext_FastAPI_QA.ipynb
└── fastapi_vectorstore/
    ├── index.faiss
    └── index.pkl
```

### File Responsibilities

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface and query pipeline |
| `utils.py` | Context formatting, citation extraction, answer formatting, and verification helpers |
| `PatchContext_FastAPI_QA.ipynb` | GitHub ingestion, document creation, chunking, indexing, and testing |
| `config.pkl` | Saved model and index configuration |
| `fastapi_vectorstore/` | Persisted FAISS vector index |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Prevents credentials and local files from being committed |

## Installation

Python 3.10 or 3.11 is recommended.

Clone the repository:

```bash
git clone https://github.com/your-username/patchcontext.git
cd patchcontext
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Upgrade pip and install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Variables

PatchContext requires a Groq API key for answer generation.

A GitHub token is only required when fetching repository data and rebuilding the index.

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=gsk_your_groq_api_key
GITHUB_TOKEN=github_pat_your_github_token
```

Create a Groq API key at:

- [Groq API Keys](https://console.groq.com/keys)

Create a GitHub token at:

- [GitHub Personal Access Tokens](https://github.com/settings/tokens)

Never commit `.env` or place real API keys directly inside the notebook or `app.py`.

## Running the Application

Make sure the following artifacts are present:

```text
config.pkl
fastapi_vectorstore/index.faiss
fastapi_vectorstore/index.pkl
```

Start the application:

```bash
streamlit run app.py
```

Do not run it with:

```bash
python app.py
```

Streamlit applications require the `streamlit run` command.

The local application will normally be available at:

```text
http://localhost:8501
```

## Building the FAISS Index

If `config.pkl` and `fastapi_vectorstore/` already exist, the app can load them directly and this step can be skipped.

To rebuild the index, open:

```text
PatchContext_FastAPI_QA.ipynb
```

Run the notebook from top to bottom. It will:

1. Connect to the GitHub API
2. Fetch FastAPI commits, pull requests, and issues
3. Convert each record into a LangChain document
4. Add source metadata to every document
5. Split the documents into chunks
6. Generate local embeddings in batches
7. Build the FAISS vector index
8. Save the index locally
9. Save the application configuration to `config.pkl`

The notebook can request credentials securely with:

```python
import os
from getpass import getpass

os.environ["GROQ_API_KEY"] = getpass("Groq API key: ")
os.environ["GITHUB_TOKEN"] = getpass("GitHub token: ")
```

The entered values are kept in memory and are not saved in the notebook.

## Saved Document Metadata

Each indexed document contains metadata that identifies its original GitHub source.

### Commit

```python
{
    "type": "commit",
    "sha": "full_commit_sha",
    "short_sha": "abc1234",
    "url": "https://github.com/fastapi/fastapi/commit/..."
}
```

### Pull Request

```python
{
    "type": "pull_request",
    "pr": 1234,
    "title": "Pull request title",
    "url": "https://github.com/fastapi/fastapi/pull/1234"
}
```

### Issue

```python
{
    "type": "issue",
    "issue": 456,
    "title": "Issue title",
    "url": "https://github.com/fastapi/fastapi/issues/456"
}
```

This metadata is used to display clickable supporting evidence.

## Query Pipeline

When a question is submitted, PatchContext performs the following steps:

1. Converts the question into an embedding using the local embedding model
2. Searches the FAISS index for similar chunks
3. Uses MMR to select relevant and diverse results
4. Uses a CrossEncoder to rerank the retrieved chunks
5. Formats the highest-ranked chunks as context
6. Sends only the selected context to Groq
7. Instructs the model to answer only from retrieved evidence
8. Extracts citations from the retrieved document metadata
9. Calculates a verification score
10. Displays the answer, warnings, and supporting GitHub links

## Grounding Rules

The generation prompt instructs the model to:

- Answer only from retrieved repository evidence
- Never fabricate commit SHAs
- Never fabricate pull request numbers
- Never fabricate issue numbers
- Answer directly without unrelated information
- State that no direct evidence was found when the context is insufficient

The notebook also contains an optional NLI-based verification step using `facebook/bart-large-mnli`. The deployed application can use a lighter verification score to reduce memory use and startup time.

## Example Questions

Try asking:

```text
Why does FastAPI use Pydantic for validation?
```

```text
Why was dependency injection added to FastAPI?
```

```text
How does FastAPI achieve high performance?
```

```text
Why does FastAPI generate automatic API documentation?
```

```text
What is the purpose of response_model?
```

```text
Which issues discuss validation with Annotated types?
```

```text
Which pull requests changed form model validation?
```

```text
What repository evidence discusses dependency cleanup behavior?
```

## Deployment

The application is deployed on Streamlit Community Cloud:

- [https://patchcontext-rag1-aryan-dadhich.streamlit.app/](https://patchcontext-rag1-aryan-dadhich.streamlit.app/)

To deploy your own version:

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select the GitHub repository.
4. Choose `app.py` as the entry point.
5. Open the app settings.
6. Add the Groq API key under Secrets.
7. Deploy the application.

Use the following Streamlit secret:

```toml
GROQ_API_KEY = "gsk_your_groq_api_key"
```

It can be read in `app.py` with:

```python
groq_api_key = st.secrets["GROQ_API_KEY"]
```

For local development, the app can instead read the value from `.env`:

```python
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
```

A simple approach that supports both environments is:

```python
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    groq_api_key = st.secrets.get("GROQ_API_KEY")
```

## Requirements

A suitable `requirements.txt` is:

```text
streamlit
langchain
langchain-core
langchain-community
langchain-huggingface
langchain-groq
faiss-cpu
sentence-transformers
transformers
torch
requests
python-dotenv
tqdm
```

## `.gitignore`

A suitable `.gitignore` is:

```gitignore
.env
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
```

Do not ignore `config.pkl` or `fastapi_vectorstore/` if the deployed application loads those artifacts directly from the repository.

## Security

Before publishing the project:

- Remove hardcoded API keys from `app.py`
- Remove hardcoded API keys from notebook cells
- Clear notebook outputs
- Add `.env` to `.gitignore`
- Store deployment credentials in Streamlit Secrets
- Search the repository for `gsk_`, `ghp_`, `github_pat_`, and `sk-`
- Revoke any key that was previously committed or shown publicly

Removing a key from the latest commit is not enough if it still exists in Git history. Any exposed key should be revoked and replaced.

## Troubleshooting

### FAISS index not found

Make sure the project contains:

```text
fastapi_vectorstore/
├── index.faiss
└── index.pkl
```

Also verify that `FAISS_INDEX_PATH` inside `config.pkl` matches the actual folder name.

### Invalid Groq API key

Create a new key at the [Groq Console](https://console.groq.com/keys).

A Groq API key normally starts with:

```text
gsk_
```

Make sure the key does not contain extra spaces or quotation marks.

### Groq rate limit

Groq's free tier has request and token limits. Wait for the limit to reset or reduce the number of requests.

### Streamlit input disappears

Store chat messages in `st.session_state` and render `st.chat_input` on every Streamlit rerun. Avoid removing required state values during the same render cycle.

### Streamlit session warnings

Always start the app using:

```bash
streamlit run app.py
```

### Slow first startup

The local embedding and reranking models are downloaded from Hugging Face during the first startup. Later runs are faster when the models are cached.

### Different embedding model

The query-time embedding model must match the model used to create the FAISS index.

If the embedding model changes, rebuild the index.

## Current Limitations

- The index contains a selected portion of FastAPI's repository history
- New GitHub activity is not included until the index is rebuilt
- The current pipeline focuses mainly on commit messages, PR descriptions, and issue bodies
- Full issue comment threads are not indexed in the current version
- Complete commit diffs are not indexed in the current version
- The verification score is a useful signal but not a guarantee that every statement is correct
- Retrieved sources should still be inspected when making important engineering decisions
- Groq model names, limits, and free-tier availability may change

## Future Improvements

- Add pull request comments
- Add issue comments
- Add truncated commit diffs
- Add incremental index updates
- Add source and date filters
- Add claim-level NLI verification
- Add RAGAs evaluation
- Add retrieval debugging to the Streamlit interface
- Add GitHub Actions for automatic index refresh
- Support repositories other than FastAPI
- Add a repository selector
- Add conversational follow-up questions

## Disclaimer

PatchContext is an independent educational project. It is not affiliated with or maintained by FastAPI, Groq, LangChain, or GitHub.

The indexed FastAPI repository content belongs to its respective authors and contributors.
