#!/usr/bin/env python3
"""
RAG Chat Application

Uses the data/ folder directly for documents. No API server.
Loads RAG and LLM in the same process and runs an interactive chat loop.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_service import create_rag_service
from llm_service import create_llm_service


def main():
    """Run the interactive RAG chat loop."""
    data_path = os.environ.get("AI_ADVISOR_DATA_PATH", "data")

    print("Loading RAG from", data_path, "...")
    rag = create_rag_service(data_path=data_path)
#     rag = create_rag_service(sources=[
#     {"url": "http://127.0.0.1:8000/items", "method": "GET"}
# ])
    llm = create_llm_service("qwen2.5:3b", backend="ollama")

    print("Ready. Commands: /stats  /search <query>  /help  quit")
    print()

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in ("quit", "exit", "q"):
                break

            if not question:
                continue

            # Handle commands
            if question == "/stats":
                print(rag.get_rag_stats())
                continue

            if question.startswith("/search "):
                query = question[8:].strip()
                context = rag.retrieve_relevant_context(query, top_k=5)
                if len(context) > 600:
                    print(context[:600], "...")
                else:
                    print(context or "(no context)")
                continue

            if question == "/help":
                print("/stats  - Show RAG statistics")
                print("/search <query>  - Search documents")
                print("quit  - Exit")
                continue

            # Normal question: RAG + LLM
            enhanced_context, status = rag.enhance_prompt_with_rag(question, top_k=5)
            prompt = (
                "Answer from the context only. If not in context say so.\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{enhanced_context}\n\n"
                "Answer:"
            )
            answer = llm.generate_answer(prompt, question)
            print("Bot:", answer)
            print("[", status, "]")

        except KeyboardInterrupt:
            break

    print("Bye.")


if __name__ == "__main__":
    main()
