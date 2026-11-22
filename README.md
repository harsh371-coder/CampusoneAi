# CampusOne AI

**A document-grounded, multi-domain question-answering backend for Bennett University information.**

CampusOne AI is a Python/FastAPI prototype that receives a student's natural-language question, identifies relevant university knowledge domains, retrieves matching document excerpts, and uses a Groq-hosted language model to produce an answer based on that evidence. It can return source references, ask for clarification, or decline out-of-scope questions.

**What it is:** an information-retrieval and question-answering assistant powered by Retrieval-Augmented Generation (RAG).  
**What it is not:** a university ERP, student-record system, transaction processor, or autonomous helpdesk that resets accounts, collects fees, or files complaints.

> **Current state:** The combined backend exists on the `feature/integration` branch. Its API and component tests have not been independently verified end to end in a live environment. The committed knowledge base is narrow; see [Current knowledge coverage](#current-knowledge-coverage).

## What happens when someone asks a question?

```text
Student question
       |
       v
POST /chat (FastAPI)
       |
       v
IntentRouter (Groq structured output; keyword fallback on failure)
       |
       +-- out of scope / clarification --> response
       |
       v
Domain-specific retrieval (Sentence Transformers + FAISS)
       |
       +-- no matching evidence -------> insufficient-evidence response
       |
       v
GroqDomainAgent for each relevant domain
       |
       v
Answer + matched document references + routing metadata
```

1. **Routing:** `IntentRouter` classifies the request into one or more supported domains, detects an intent, and reports a routing confidence score. If the Groq routing call fails, the code uses deterministic keyword rules.
2. **Retrieval:** `RetrievalService` searches local, domain-specific FAISS indexes. The embedding model is `sentence-transformers/all-MiniLM-L6-v2`; retrieval currently defaults to `top_k=5` and a minimum similarity score of `0.25`.
3. **Answer generation:** Each selected domain uses an instance of `GroqDomainAgent` with its own system prompt. The agent is instructed to use retrieved university evidence and acknowledge missing information.
4. **Output:** The API combines nonempty domain answers, attempts to match the agent's sources to retrieved chunks, and returns response flags and metadata.

Multi-domain questions can produce sections labelled by domain. The current implementation uses the same underlying agent class and configured model for all four domains; these are **logical domain agents**, not separately deployed services.

## Supported routing domains

The domain labels describe **what the system can classify and is designed to retrieve**, not proof that corresponding reference documents are already present.

| Code label | Intended questions |
| --- | --- |
| `IT` | Wi-Fi, university email, student portal access, login/password issues, software support |
| `HR_ADMIN` | Student academic rules, examinations, attendance, grades/CGPA, course registration, academic documents and administrative procedures |
| `FEES` | Fee structures, payment procedures, deadlines, refunds and fee-related policies |
| `FACILITIES` | Hostels, transportation, maintenance, campus infrastructure and student services |

**Naming note:** `HR_ADMIN` is a legacy code identifier. In this implementation it primarily represents **academic/student administration**, not a populated human-resources knowledge base.

## Current knowledge coverage

The committed dataset on `feature/integration` is:

`data/processed/scset_ug_chunks.jsonl`

It contains **23 processed chunks from one document**, *Academic Rules for SCSET UG*, sourced from Bennett University. All 23 chunks carry `category: academic_rules` and are mapped to `HR_ADMIN` by `backend/rag/retrieval/dataset_loader.py`.

The source content includes degree/semester and course-registration rules, examination and supplementary/makeup-exam regulations, grading and CGPA, attendance-related rules, internships, and honors/minors requirements.

**Limitations of the committed data:**

- No dedicated IT, fee-policy, or facilities knowledge chunks are committed in this dataset.
- Some nonacademic topics appear only incidentally inside academic rules; this does not amount to full coverage of those domains.
- No explicit revaluation procedure was found in these 23 chunks.
- The extracted text includes occasional broken words and section boundaries; retrieval relevance and answer quality need evaluation.
- FAISS indexes are created locally or loaded from `data/processed/vector_stores/`; their presence and completeness depend on the running environment.

The system **should not be assumed capable of reliably answering all four domains** until suitable, verified documents are indexed and responses are tested. Even an in-domain question may lack supporting evidence.

## Tech stack

- **Backend:** Python, FastAPI, Pydantic; Uvicorn for serving
- **LLM provider:** Groq (router and answer generation)
- **Default configured model:** `openai/gpt-oss-20b`
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector search/storage:** local FAISS indexes with JSON metadata
- **Data format:** processed JSON/JSONL document chunks
- **Tests:** pytest, unittest mocks, and local manual test scripts

There is **no application SQL database, authentication service, or conversation-history store** implemented in the examined backend.

## Repository structure

```text
backend/
  main.py                     # FastAPI application; /, /health, /chat
  router/
    router.py                 # Groq intent/domain classification
    fallback.py               # Keyword rules and routing policy
    labels.py                 # Domains and intents
    schemas.py                # Routing response models
  rag/retrieval/
    dataset_loader.py         # Maps JSONL chunk categories to domains
    embeddings.py             # Sentence Transformer embeddings
    vector_store.py           # FAISS index and metadata persistence
    retriever.py              # Domain-aware semantic retrieval
    service.py                # RetrievalService wrapper
  agents/
    base_agent.py             # Shared agent interface/response
    groq_agent.py             # Groq answer generation and structured output
    prompts.py                # Grounding rules and domain prompts
  schemas/api.py              # Chat request/response models

data/processed/
  scset_ug_chunks.jsonl       # Currently committed knowledge chunks
build_real_index.py           # Optional index-building script
tests/                        # Automated tests
test_*.py                     # Additional local test/manual scripts
requirements.txt
.env.example
```

Some additional directories in the repository are scaffolding and are not implemented services.

## Running the backend locally

Run commands from the **repository root** on the `feature/integration` branch.

**Prerequisites:** a compatible Python 3 environment, network access to install dependencies and obtain the embedding model on first use, and a valid Groq API key.

```bash
git clone https://github.com/harsh371-coder/CampusoneAi.git
cd CampusoneAi
git checkout feature/integration

python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set at least:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_ROUTER_MODEL=openai/gpt-oss-20b
```

Keep `.env` and API keys out of version control.

Start the API:

```bash
uvicorn backend.main:app --reload
```

On import, the backend initializes the embedding service and attempts to load existing FAISS indexes or build them from the committed JSONL file. **Note:** the current code checks whether *any* domain index exists before loading, so incomplete persisted indexes can result in partially empty domains; remove/rebuild indexes when updating the dataset.

Optionally build indexes explicitly:

```bash
python build_real_index.py
```

FastAPI exposes interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

## API reference

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/` | Returns a short running-status message |
| `GET` | `/health` | Returns `{"status": "ok"}`; does not independently test all dependencies |
| `POST` | `/chat` | Routes, retrieves, and generates a document-grounded response |

### Chat request

```json
{
  "message": "What are the rules for a makeup examination?",
  "session_id": "optional-client-session-id"
}
```

- `message` is required (1–2000 characters by the Pydantic schema).
- `session_id` is optional and currently **passed through only**; it does not enable stored conversation memory.

### Chat response schema (illustrative shape, not a recorded model answer)

```json
{
  "answer": "Generated answer or an explanation that verified information is unavailable.",
  "domains": ["HR_ADMIN"],
  "confidence": 0.85,
  "needs_clarification": false,
  "clarification_question": null,
  "is_out_of_scope": false,
  "sources": [
    {
      "chunk_id": "scset_ug_0011",
      "domain": "HR_ADMIN",
      "source_title": "Academic Rules for SCSET UG",
      "page": 9,
      "score": 0.65,
      "source_url": "https://www.bennett.edu.in/wp-content/uploads/2022/03/Academic-Rules-for-SCSET_UG.pdf"
    }
  ],
  "session_id": "optional-client-session-id",
  "metadata": {
    "routing_method": "llm",
    "intent": "exam_rules",
    "agent_confidences": {
      "HR_ADMIN": 0.8
    },
    "retrieved_chunks": 5,
    "status": "answered"
  }
}
```

**Important:** These numbers, citations, and answer content are examples of field shape only, not verified output from a live request. `confidence` is the **routing** confidence, not measured answer correctness. `agent_confidences` are model-reported confidence values, not calibrated reliability scores. The `sources` array may be empty.

Common `metadata.status` values include `answered`, `out_of_scope`, `needs_clarification`, `no_retrieved_evidence`, and `provider_rate_limited`. The rate-limit case is returned as a response body, while unexpected errors can return HTTP 500.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the rules for a makeup examination?"}'
```

## Tests and verification

```bash
pytest tests/
```

Additional root-level `test_*.py` files include manual entry-point scripts and retrieval diagnostics; not all are formal pytest test cases. Some agent tests mock Groq responses rather than exercising a real provider call. Running tests and a live `/chat` request is necessary to validate the environment and end-to-end behavior.

## Known limitations

- **Evidence coverage:** currently centered on one SCSET undergraduate academic-rules document.
- **No conversation persistence:** `session_id` does not load or save message history; the chat handler passes an empty conversation context to agents.
- **Citation limitations:** model-produced source references are matched to retrieved chunks, but successful matching and claim-level faithfulness are not guaranteed.
- **Retrieval quality:** the similarity cutoff is permissive and does not guarantee that a retrieved chunk answers the question.
- **Index lifecycle:** startup index detection does not validate that every expected domain index exists or is current.
- **No transactional actions:** no fee payment, account reset, grievance submission, or university-record modification.
- **No user authentication or authorization:** do not treat the current API as a student identity system.
- **Prototype operations:** permissive CORS, no API-level user rate limiting, and limited deployment/observability setup.
- **Evaluation not established:** the repository includes tests, but does not establish independently measured production routing accuracy or end-to-end answer accuracy.

## Scope and development status

The current implementation is a **backend prototype for university document Q&A** with modular routing, retrieval, and domain prompts. The next substantive backend steps are to validate live end-to-end behavior, strengthen evidence/citation reliability, and add current authoritative documents for the remaining domains.

This README describes the code and data currently present on `feature/integration`, not a future roadmap or unverified product capabilities.
