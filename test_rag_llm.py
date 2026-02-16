#!/usr/bin/env python3
"""
Test script for RAG and LLM.

Run: python test_rag_llm.py

Checks:
  1. Text loader (loads documents from data folder)
  2. RAG service (embeddings and retrieval)
  3. LLM (Ollama by default; must be running)
"""

import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))


def test_text_loader() -> bool:
    """Test that the text loader can load documents from the data folder."""
    print("=" * 50)
    print("1. TEXT LOADER (data folder)")
    print("=" * 50)

    from text_loader import load_path, get_supported_extensions

    data_path = os.environ.get("AI_ADVISOR_DATA_PATH", "data")
    if not Path(data_path).exists():
        print(f"   SKIP: folder '{data_path}' not found")
        return False

    documents = load_path(data_path)
    extensions = get_supported_extensions()

    print(f"   Supported extensions: {len(extensions)} types")
    print(f"   Documents loaded from '{data_path}': {len(documents)}")

    if not documents:
        print("   WARNING: No documents loaded. Add PDF/TXT/DOCX etc. to data/")
        return False

    for i, doc in enumerate(documents[:3]):
        filename = doc.get("metadata", {}).get("filename", "?")
        content_len = len(doc.get("content", ""))
        print(f"   - {filename}: {content_len} chars")

    print("   PASS: Text loader OK")
    print()
    return True


def test_rag() -> bool:
    """Test RAG service: load docs, build embeddings, retrieve context."""
    print("=" * 50)
    print("2. RAG SERVICE (embed + retrieve)")
    print("=" * 50)

    from rag_service import create_rag_service

    data_path = os.environ.get("AI_ADVISOR_DATA_PATH", "data")
    rag = create_rag_service(data_path=data_path)
    stats = rag.get_rag_stats()

    print(f"   Embedding model loaded: {stats.get('embedding_model_loaded')}")
    print(f"   Cached embeddings: {stats.get('cached_embeddings')}")
    print(f"   Documents: {stats.get('documents')}")

    if not stats.get("cached_embeddings"):
        print("   WARNING: No embeddings (no docs in data/ or model failed)")
        return False

    context = rag.retrieve_relevant_context(
        "what is this document about?",
        top_k=2,
    )
    print(f"   Retrieve context length: {len(context)} chars")
    if context:
        print(f"   Sample: {context[:200].strip()}...")

    print("   PASS: RAG OK")
    print()
    return True


def test_llm() -> bool:
    """Test LLM (Ollama). Fails if Ollama is not running."""
    print("=" * 50)
    print("3. LLM (Ollama default)")
    print("=" * 50)

    from llm_service import create_llm_service

    try:
        llm = create_llm_service("qwen2.5:3b", backend="ollama")
        answer = llm.generate_answer("Say only: Hello RAG.", "test")

        if "LLM error" in answer or "Connection" in answer or "HTTP" in answer:
            print(f"   FAIL: {answer}")
            print("   Make sure Ollama is running: ollama serve && ollama pull qwen2.5:3b")
            return False

        print(f"   Response: {answer[:150]}...")
        print("   PASS: LLM OK")
        print()
        return True

    except Exception as e:
        print(f"   FAIL: {e}")
        print("   Start Ollama: ollama serve  and  ollama pull qwen2.5:3b")
        return False


def main() -> int:
    """Run all tests and print summary."""
    print()
    print("RAG + LLM Project Test")
    print()

    result_loader = test_text_loader()
    result_rag = test_rag()
    result_llm = test_llm()

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"   Text loader: {'PASS' if result_loader else 'FAIL'}")
    print(f"   RAG:         {'PASS' if result_rag else 'FAIL'}")
    print(f"   LLM:         {'PASS' if result_llm else 'FAIL'}")
    print()

    if result_loader and result_rag and result_llm:
        print("All OK. Your RAG and LLM are working.")
        return 0
    print("Fix failures above, then run again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
