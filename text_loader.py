#!/usr/bin/env python3
"""
All-in-one Loaders for RAG

Single file with every loader type:
- PDF, DOCX, TXT/MD (document)
- HTML, JSON, CSV (structured)
- Plain text (all other extensions)
- API loader (fetch from URL / REST API)

Returns list of {content, metadata} for RAG.
"""

import os
import re
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Optional deps for document types
try:
    import PyPDF2
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# -----------------------------------------------------------------------------
# Supported extensions
# -----------------------------------------------------------------------------

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv",
    ".json", ".xml", ".yaml", ".yml", ".ini", ".cfg", ".conf",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".sql", ".sh", ".bat", ".ps1",
    ".html", ".htm", ".xhtml", ".css", ".scss", ".vue", ".r", ".rb",
    ".go", ".rs", ".java", ".kt", ".c", ".cpp", ".h", ".hpp",
    ".php", ".asp", ".aspx", ".jsp", ".cs", ".vb", ".swift",
    ".tex", ".bib", ".nfo", ".srt", ".vtt", ".env", ".gitignore",
}

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc"}

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _read_text_file(
    path: str,
    encodings: tuple = ("utf-8", "latin-1", "cp1252"),
) -> Optional[str]:
    """Read a text file with encoding fallback."""
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, OSError):
            continue
    return None


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return text.strip()


def _json_to_text(obj: Any) -> str:
    """Flatten JSON to readable text for RAG."""
    if isinstance(obj, dict):
        return " | ".join(f"{k}: {_json_to_text(v)}" for k, v in obj.items())
    if isinstance(obj, list):
        return " | ".join(_json_to_text(item) for item in obj)
    return str(obj)


# -----------------------------------------------------------------------------
# PDF Loader
# -----------------------------------------------------------------------------

def _load_pdf(file_path: str) -> Optional[str]:
    """Extract full text from PDF. Returns None on failure."""
    if not HAS_PDF:
        return None
    try:
        with pdfplumber.open(file_path) as pdf:
            parts = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t.strip())
            return "\n".join(parts) if parts else None
    except Exception:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                parts = []
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t.strip())
                return "\n".join(parts) if parts else None
        except Exception:
            return None


# -----------------------------------------------------------------------------
# DOCX Loader
# -----------------------------------------------------------------------------

def _load_docx(file_path: str) -> Optional[str]:
    """Extract text from DOCX. Returns None on failure."""
    if not HAS_DOCX:
        return None
    try:
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# TXT / MD Loader
# -----------------------------------------------------------------------------

def _load_txt(file_path: str) -> Optional[str]:
    """Read plain text file with encoding fallback."""
    return _read_text_file(file_path)


# -----------------------------------------------------------------------------
# HTML Loader
# -----------------------------------------------------------------------------

def _load_html(file_path: str) -> Optional[str]:
    """Read HTML and return stripped text."""
    raw = _read_text_file(file_path)
    return _strip_html(raw) if raw else None


# -----------------------------------------------------------------------------
# JSON Loader
# -----------------------------------------------------------------------------

def _load_json(file_path: str) -> Optional[str]:
    """Read JSON and flatten to text."""
    raw = _read_text_file(file_path)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return _json_to_text(obj)
    except json.JSONDecodeError:
        return raw


# -----------------------------------------------------------------------------
# CSV Loader
# -----------------------------------------------------------------------------

def _load_csv(file_path: str) -> Optional[str]:
    """Read CSV and return table as text lines."""
    raw = _read_text_file(file_path)
    if not raw:
        return None
    try:
        rows = list(csv.reader(raw.splitlines()))
        return "\n".join(" | ".join(cell for cell in row) for row in rows)
    except Exception:
        return raw


# -----------------------------------------------------------------------------
# API Loader
# -----------------------------------------------------------------------------

def load_from_api(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Load content from a URL / REST API.

    - method: "GET" or "POST"
    - headers: optional dict of request headers
    - body: optional dict for JSON body (used with POST)
    - Returns list of one document with 'content' and 'metadata'.
      If response is JSON, it is flattened to text; otherwise raw text is used.
    """
    if not HAS_REQUESTS:
        return []

    metadata = {
        "source": url,
        "filename": url.split("?")[0].rstrip("/").split("/")[-1] or "api",
        "extension": ".api",
        "loader": "api",
    }

    try:
        h = headers or {}
        if method.upper() == "GET":
            response = requests.get(url, headers=h, timeout=timeout)
        else:
            h.setdefault("Content-Type", "application/json")
            response = requests.post(
                url, headers=h, json=body, timeout=timeout
            )

        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()

        if "json" in content_type:
            try:
                obj = response.json()
                text = _json_to_text(obj)
            except Exception:
                text = response.text
        else:
            text = response.text

        if not (text and text.strip()):
            return []
        return [{"content": text.strip(), "metadata": metadata}]
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Single file loader (dispatches by extension)
# -----------------------------------------------------------------------------

def load_text_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a single file by extension. Uses the appropriate loader.
    Returns list of {content, metadata} dicts.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return []

    ext = path.suffix.lower()
    filename = path.name
    metadata = {
        "source": str(path),
        "filename": filename,
        "extension": ext,
    }

    content = None

    if ext == ".pdf":
        content = _load_pdf(file_path)
    elif ext in (".docx", ".doc"):
        content = _load_docx(file_path)
    elif ext in (".txt", ".md", ".markdown"):
        content = _load_txt(file_path)
        if not content and path.exists():
            content = _read_text_file(file_path)
    elif ext in (".html", ".htm", ".xhtml"):
        content = _load_html(file_path)
    elif ext == ".json":
        content = _load_json(file_path)
    elif ext == ".csv":
        content = _load_csv(file_path)
    elif ext in TEXT_EXTENSIONS:
        content = _read_text_file(file_path)

    if content is None or not content.strip():
        return []
    return [{"content": content.strip(), "metadata": metadata}]


# -----------------------------------------------------------------------------
# Directory loader
# -----------------------------------------------------------------------------

def load_directory(
    directory: str,
    recursive: bool = True,
    extensions: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Load all supported files from a directory."""
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        return []

    exts = extensions or SUPPORTED_EXTENSIONS
    documents = []
    it = path.rglob("*") if recursive else path.iterdir()
    for fp in it:
        if fp.is_file() and fp.suffix.lower() in exts:
            documents.extend(load_text_file(str(fp)))
    return documents


# -----------------------------------------------------------------------------
# Unified entry points
# -----------------------------------------------------------------------------

def load_path(
    path: str,
    recursive: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load from a local path (file or directory).
    Main entry point for the RAG system.
    """
    p = Path(path)
    if not p.exists():
        return []
    if p.is_file():
        return load_text_file(str(p))
    return load_directory(str(p), recursive=recursive)


def load_sources(
    sources: List[Any],
    recursive: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load from a list of sources. Each source can be:
    - str path (file or directory)
    - dict with "url" and optional "method", "headers", "body" for API
    """
    documents = []
    for src in sources:
        if isinstance(src, dict):
            url = src.get("url")
            if not url:
                continue
            docs = load_from_api(
                url=url,
                method=src.get("method", "GET"),
                headers=src.get("headers"),
                body=src.get("body"),
            )
            documents.extend(docs)
        elif isinstance(src, str):
            if src.startswith("http://") or src.startswith("https://"):
                documents.extend(load_from_api(src))
            else:
                documents.extend(load_path(src, recursive=recursive))
    return documents


def get_supported_extensions() -> set:
    """Return the set of supported file extensions."""
    return SUPPORTED_EXTENSIONS


# -----------------------------------------------------------------------------
# Backward compatibility: extract_* for file_processor callers
# -----------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> Optional[Dict[str, Any]]:
    """Extract text from PDF. Returns dict with 'full_text' and metadata."""
    text = _load_pdf(pdf_path)
    if not text:
        return None
    return {
        "file_name": os.path.basename(pdf_path),
        "file_path": pdf_path,
        "file_size": os.path.getsize(pdf_path),
        "pages": [],
        "full_text": text,
        "extraction_timestamp": datetime.now().isoformat(),
    }


def extract_text_from_docx(docx_path: str) -> Optional[Dict[str, Any]]:
    """Extract text from DOCX. Returns dict with 'full_text' and metadata."""
    text = _load_docx(docx_path)
    if not text:
        return None
    return {
        "file_name": os.path.basename(docx_path),
        "file_path": docx_path,
        "file_size": os.path.getsize(docx_path),
        "pages": [{"page_number": 1, "text": text.strip()}],
        "full_text": text.strip(),
        "extraction_timestamp": datetime.now().isoformat(),
    }


def extract_text_from_txt(txt_path: str) -> Optional[Dict[str, Any]]:
    """Extract text from TXT/MD. Returns dict with 'full_text' and metadata."""
    text = _load_txt(txt_path)
    if not text:
        return None
    return {
        "file_name": os.path.basename(txt_path),
        "file_path": txt_path,
        "file_size": os.path.getsize(txt_path),
        "pages": [{"page_number": 1, "text": text.strip()}],
        "full_text": text.strip(),
        "extraction_timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data"
    docs = load_path(path)
    print(f"Loaded {len(docs)} document(s) from {path}")
    # Example API load (uncomment and set URL to test):
    # api_docs = load_from_api("https://api.example.com/data")
    # print("API docs:", len(api_docs))
