# RAG Advisor – Full Code Explanation

This document explains how your whole project works, file by file and flow by flow.

---

## 1. What the Project Does

You have a **RAG (Retrieval-Augmented Generation) chat app** that:

1. **Loads** documents from a folder (and optionally from APIs).
2. **Indexes** them with embeddings (vector representations) so you can search by meaning.
3. **Answers** your questions by: finding relevant chunks → building a prompt → calling an LLM (Ollama/OpenAI/llama.cpp).

No database, no API server, no API keys. One command: `python secure_app.py`.

---

## 2. High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER runs: python secure_app.py                  │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  secure_app.py                                                           │
│  • Reads AI_ADVISOR_DATA_PATH (default: "data")                          │
│  • Creates RAG service (loads docs + builds embeddings)                   │
│  • Creates LLM service (e.g. Ollama qwen2.5:3b)                          │
│  • Starts chat loop: input → RAG context → LLM → print answer            │
└─────────────────────────────────────────────────────────────────────────┘
         │                                    │
         │                                    │
         ▼                                    ▼
┌─────────────────────┐            ┌─────────────────────┐
│  rag_service.py     │            │  llm_service.py      │
│  • load_path() or   │            │  • Ollama / OpenAI  │
│    load_sources()   │            │    / llama.cpp      │
│  • Chunk docs       │            │  • generate_answer()│
│  • Embed (MiniLM)   │            └─────────────────────┘
│  • retrieve_relevant│
│    _context()       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  text_loader.py     │
│  • PDF, DOCX, TXT   │
│  • HTML, JSON, CSV  │
│  • Plain text       │
│  • API (URL)        │
└─────────────────────┘
```

---

## 3. File-by-File Explanation

### 3.1 `secure_app.py` – Entry Point & Chat Loop

**Role:** The main script you run. It wires RAG + LLM and runs the interactive chat.

**What it does:**

1. **Setup**
   - Gets `data_path` from env `AI_ADVISOR_DATA_PATH` (default `"data"`).
   - Creates `RAGService(data_path=data_path)` → loads documents and builds embeddings.
   - Creates `LLMService("qwen2.5:3b", backend="ollama")` → ready to call Ollama.

2. **Loop**
   - Reads a line from the user (`input("\nYou: ")`).
   - **Commands:**
     - `quit` / `exit` / `q` → exit.
     - `/stats` → prints RAG stats (embedding count, document count, etc.).
     - `/search <query>` → retrieves relevant context for the query (no LLM), prints it.
     - `/help` → prints command list.
   - **Normal question:**
     - Calls `rag.enhance_prompt_with_rag(question, top_k=5)` → gets relevant document text.
     - Builds a prompt: “Answer from the context only. Question: … Context: …”.
     - Calls `llm.generate_answer(prompt, question)` → gets the model’s answer.
     - Prints the answer and RAG status (e.g. “RAG-enhanced”).

So: **secure_app.py = load RAG + LLM once, then for each question: RAG retrieval → prompt → LLM → print.**

---

### 3.2 `rag_service.py` – RAG (Retrieval + Embeddings)

**Role:** Load documents, turn them into vectors, and retrieve the most relevant chunks for a question.

**Main ideas:**

- **Documents** come from `text_loader` (folder and/or API). Each doc is `{ "content": "...", "metadata": {...} }`.
- **Chunking:** Long content is split into chunks of 400 characters with 100-character overlap so retrieval is at chunk level.
- **Embeddings:** Each chunk (and the user question) is turned into a vector using **SentenceTransformer** (`all-MiniLM-L6-v2`). Vectors are stored in memory (no database).
- **Retrieval:** For a question, its vector is compared to all chunk vectors with **cosine similarity**; top‑k chunks are returned as “context” for the LLM.

**Key methods:**

| Method | What it does |
|--------|------------------|
| `__init__(data_path, sources)` | If `sources` is given, uses `load_sources(sources)`; else uses `load_path(data_path)`. Then loads embedding model and runs `_load_and_embed()`. |
| `_load_and_embed()` | Gets list of docs from loader → chunks long docs → builds one text per chunk → encodes all with the model → stores in `embeddings_cache` (texts, embeddings, records). |
| `retrieve_relevant_context(question, top_k)` | Encodes the question → cosine similarity with all chunk embeddings → picks top_k → returns concatenated text of those chunks. |
| `enhance_prompt_with_rag(question, top_k)` | Calls `retrieve_relevant_context`, then returns (1) a string “Document context: …” + that context, (2) status like “RAG-enhanced” or “No RAG context”. |
| `get_rag_stats()` | Returns dict: embedding model loaded, number of cached embeddings, number of documents, data_path. |

So: **rag_service = load docs → chunk → embed → on each question, return top‑k chunks as context.**

---

### 3.3 `text_loader.py` – All Loaders in One File

**Role:** Turn files and APIs into a single format: list of `{ "content": str, "metadata": {...} }` for RAG.

**Loader types:**

| Type | Function | Description |
|------|----------|-------------|
| **PDF** | `_load_pdf()` | pdfplumber first, fallback PyPDF2. Concatenates all page text. |
| **DOCX** | `_load_docx()` | python-docx: all paragraphs joined. |
| **TXT/MD** | `_load_txt()` | Read file with encoding fallback (utf-8, latin-1, cp1252). |
| **HTML** | `_load_html()` | Read file, strip tags, normalize entities. |
| **JSON** | `_load_json()` | Parse JSON, flatten to one text string (keys and values). |
| **CSV** | `_load_csv()` | Parse CSV, each row becomes one line of text (cells joined). |
| **Plain text** | `_read_text_file()` | Any other supported extension (.py, .log, .yaml, etc.) read as text. |
| **API** | `load_from_api()` | HTTP GET or POST to a URL; if response is JSON, flatten to text; else use body as text. |

**Entry points:**

- **`load_path(path, recursive=True)`**  
  - If `path` is a file → `load_text_file(path)`.  
  - If directory → `load_directory(path, recursive)` (all supported extensions).  
  - Used by RAG when you only pass a folder (e.g. `data`).

- **`load_sources(sources, recursive=True)`**  
  - `sources`: list of:
    - **String path** → same as `load_path` (file or directory).
    - **URL string** (http/https) → `load_from_api(url)`.
    - **Dict** `{"url": "...", "method": "POST", "headers": {...}, "body": {...}}` → full API call.
  - Returns one list of documents from all sources. Used by RAG when you pass `sources= [...]`.

**`load_text_file(file_path)`**  
Dispatches by extension (`.pdf`, `.docx`, `.txt`, `.html`, `.json`, `.csv`, or any in `TEXT_EXTENSIONS`) and calls the right internal loader. Returns `[{ "content": "...", "metadata": { "source", "filename", "extension" } }]`.

**Backward compatibility:**  
`extract_text_from_pdf`, `extract_text_from_docx`, `extract_text_from_txt` return the old-style dict with `full_text` and metadata so `file_processor` (and any old code) still works.

So: **text_loader = one place for every input type (files + API) and one document format for RAG.**

---

### 3.4 `llm_service.py` – Language Model Calls

**Role:** Send a prompt to an LLM and return the generated text.

**Backends:**

| Backend | URL | How it’s called |
|---------|-----|-------------------|
| **ollama** | http://localhost:11434 | POST `/api/generate` with `model`, `prompt`, `stream: false`. |
| **llamacpp** | http://localhost:8080 | POST `/completion` with `prompt`, `n_predict`, `temperature`. |
| **openai** | https://api.openai.com/v1 | POST `/chat/completions` with `OPENAI_API_KEY` in header. |

**Flow:**

- `create_llm_service(model_name, backend="ollama")` → builds `LLMService`.
- `generate_answer(prompt, question)` → calls the right private method (`_ollama`, `_llamacpp`, `_openai`), returns the response text or an error string (e.g. connection failed, HTTP error).

So: **llm_service = single interface to Ollama / OpenAI / llama.cpp.**

---

### 3.5 `file_processor.py` – Compatibility Layer

**Role:** Keep old imports working. All real logic is in `text_loader.py`.

It only re-exports:

- `extract_text_from_pdf`
- `extract_text_from_docx`
- `extract_text_from_txt`

So any code that does `from file_processor import extract_text_from_pdf` still works. The `__main__` block loads from a path (default `data`) using `load_path` and prints how many documents were loaded.

---

### 3.6 `test_rag_llm.py` – Test Script

**Role:** Check that loader, RAG, and LLM all work.

**Three tests:**

1. **Text loader**  
   Uses `load_path(data_folder)` and `get_supported_extensions()`. Checks that at least one document loads and prints a short summary.

2. **RAG**  
   Creates `create_rag_service(data_path=...)`, then checks stats and calls `retrieve_relevant_context("what is this document about?", top_k=2)` to ensure retrieval returns something.

3. **LLM**  
   Creates `create_llm_service("qwen2.5:3b", backend="ollama")` and asks for “Say only: Hello RAG.” Fails if Ollama isn’t running or returns an error.

**Run:** `python test_rag_llm.py`  
Exit code 0 = all pass; 1 = at least one failed.

---

### 3.7 `requirements.txt` – Dependencies

- **requests** – HTTP (API loader, LLM calls).
- **PyPDF2, pdfplumber** – PDF text extraction.
- **python-docx** – DOCX text extraction.
- **sentence-transformers** – embedding model (MiniLM).
- **scikit-learn** – cosine similarity for retrieval.
- **numpy** – arrays for embeddings/similarity.

Ollama/OpenAI/llama.cpp are not in requirements; they are external services you run or configure.

---

### 3.8 `README.md` – Usage and Examples

Explains:

- Run with `python secure_app.py`, documents in `data/`.
- All loaders live in `text_loader.py` (PDF, DOCX, TXT, HTML, JSON, CSV, plain text, API).
- How to use **API loader**: `load_from_api(url, ...)` and `load_sources([...])`, including mixing folder + API in `create_rag_service(sources=[...])`.
- Env vars: `AI_ADVISOR_DATA_PATH`, and LLM (Ollama vs `OPENAI_API_KEY`).

---

## 4. Data Flow for One Question

1. User types: `What are the main topics in the document?`
2. **secure_app** calls `rag.enhance_prompt_with_rag(question, top_k=5)`.
3. **rag_service**:
   - Encodes the question with the same embedding model.
   - Computes cosine similarity between question vector and all chunk vectors.
   - Takes top 5 chunks, concatenates their text.
   - Returns that as “context” plus status.
4. **secure_app** builds:
   - Prompt = “Answer from the context only. Question: … Context: <retrieved chunks>”
5. **secure_app** calls `llm.generate_answer(prompt, question)`.
6. **llm_service** sends the prompt to Ollama (or other backend), returns the model’s reply.
7. **secure_app** prints: `Bot: <answer>` and `[ RAG-enhanced ]`.

So: **Question → RAG (embed + similarity + top‑k chunks) → prompt with context → LLM → answer.**

---

## 5. Summary Table

| File | Purpose |
|------|--------|
| **secure_app.py** | Entry point; chat loop; RAG + LLM per question. |
| **rag_service.py** | Load docs (path or sources), chunk, embed, retrieve top‑k by similarity. |
| **text_loader.py** | All loaders (PDF, DOCX, TXT, HTML, JSON, CSV, text, API); one document format. |
| **llm_service.py** | Call Ollama / OpenAI / llama.cpp with a prompt. |
| **file_processor.py** | Re-export extract_* from text_loader for compatibility. |
| **test_rag_llm.py** | Test loader, RAG, and LLM. |
| **requirements.txt** | Python dependencies. |
| **README.md** | How to run and use loaders/API. |

Together, this is a single-command RAG chat app: documents from disk (and optionally from APIs), in-memory embeddings, and an LLM that answers using retrieved context.
