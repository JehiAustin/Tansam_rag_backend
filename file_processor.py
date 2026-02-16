#!/usr/bin/env python3
"""
File Processor (backward compatibility)

All extraction logic now lives in text_loader.py.
This module re-exports the same functions so existing imports keep working.
"""

from text_loader import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
)

__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "extract_text_from_txt",
]


if __name__ == "__main__":
    import sys
    from text_loader import load_path

    path = sys.argv[1] if len(sys.argv) > 1 else "data"
    documents = load_path(path)
    print(f"Loaded {len(documents)} document(s) from {path}")
