#!/usr/bin/env python3
"""
FastAPI Application for RAG Chat

HTTP API version of the RAG chat application.
RAG and LLM services are loaded once at startup and reused for all requests.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_service import create_rag_service, RAGService
from llm_service import create_llm_service, LLMService, get_optimal_model
from llm_optimizer import LLMOptimizer


# -----------------------------------------------------------------------------
# Global services (initialized at startup)
# -----------------------------------------------------------------------------

rag_service: Optional[RAGService] = None
llm_service: Optional[LLMService] = None
llm_optimizer: Optional[LLMOptimizer] = None


# -----------------------------------------------------------------------------
# Lifespan context manager for startup/shutdown
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize RAG and LLM services at startup.
    Clean up resources at shutdown if needed.
    """
    global rag_service, llm_service, llm_optimizer

    # Startup: Load services
    print("=" * 60)
    print("Initializing RAG and LLM services...")
    print("=" * 60)

    data_path = os.environ.get("AI_ADVISOR_DATA_PATH", "data")
    model_name = os.environ.get("AI_ADVISOR_MODEL_NAME", "auto")  # Use auto-detection
    backend = os.environ.get("AI_ADVISOR_LLM_BACKEND", "ollama")

    print(f"Data path: {data_path}")
    
    # Get optimal model if auto is selected
    if model_name == "auto":
        optimal_model, optimal_backend = get_optimal_model()
        print(f"Auto-detected optimal model: {optimal_model}")
        model_name = optimal_model
        backend = optimal_backend
    
    print(f"LLM model: {model_name}")
    print(f"LLM backend: {backend}")
    print()

    try:
        rag_service = create_rag_service(data_path=data_path)
        print()
        llm_service = create_llm_service(model_name=model_name, backend=backend)
        print()
        llm_optimizer = LLMOptimizer()
        print("🚀 LLM Optimizer initialized")
        print()
        print("=" * 60)
        print("Services ready. API is available.")
        print("=" * 60)
    except Exception as e:
        print(f"[ERROR] Failed to initialize services: {e}")
        raise

    yield

    # Shutdown: Cleanup (if needed)
    print("\nShutting down...")


# -----------------------------------------------------------------------------
# FastAPI app initialization
# -----------------------------------------------------------------------------

app = FastAPI(
    title="RAG Chat API",
    description="RAG-based question answering API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    question: str = Field(..., description="The question to ask", min_length=1)


class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""
    answer: str = Field(..., description="The generated answer")
    status: str = Field(..., description="RAG status: 'RAG-enhanced' or 'No RAG context'")


class StatsResponse(BaseModel):
    """Response model for /stats endpoint."""
    embedding_model_loaded: bool
    cached_embeddings: int
    documents: int
    data_path: str


# -----------------------------------------------------------------------------
# Health check endpoint
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG Chat API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "POST /chat",
            "stats": "GET /stats",
            "health": "GET /health",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_ready": rag_service is not None,
        "llm_ready": llm_service is not None,
    }


# -----------------------------------------------------------------------------
# Main endpoints
# -----------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint: Ask a question and get an answer using RAG + LLM.

    - **question**: The question to ask (required, min length 1)
    - Returns: answer and RAG status
    """
    if rag_service is None or llm_service is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized. Please check server logs.",
        )

    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Check cache first
        if llm_optimizer:
            cached_answer = llm_optimizer.get_cached_response(question)
            if cached_answer:
                return ChatResponse(answer=cached_answer, status="cached")

        # Optimize question for faster processing
        optimized_question = question
        if llm_optimizer:
            optimized_question = llm_optimizer.optimize_question(question)

        # Get RAG-enhanced context
        enhanced_context, status = rag_service.enhance_prompt_with_rag(
            optimized_question, top_k=3  # Reduced for speed
        )

        # Build shorter prompt for LLM
        prompt = (
            f"Q: {optimized_question}\n"
            f"Context: {enhanced_context[:500]}...\n"  # Limit context length
            "A:"
        )

        # Generate answer with timeout handling
        try:
            answer = llm_service.generate_answer(prompt, optimized_question)
        except Exception as llm_error:
            # Use fallback response if LLM fails
            if llm_optimizer:
                answer = llm_optimizer.get_fallback_response(question)
                status = "fallback"
            else:
                answer = f"LLM error: {llm_error}"
                status = "error"

        # Cache successful responses
        if llm_optimizer and "error" not in status:
            llm_optimizer.cache_response(question, answer)

        return ChatResponse(answer=answer, status=status)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}",
        )


@app.get("/chat")
async def chat_get():
    """
    GET endpoint for chat - provides usage instructions and simple form.
    """
    return {
        "message": "Use POST method to ask questions",
        "usage": {
            "method": "POST",
            "url": "/chat",
            "headers": {"Content-Type": "application/json"},
            "body": {"question": "Your question here"},
            "example": "curl -X POST http://127.0.0.1:8000/chat -H \"Content-Type: application/json\" -d '{\"question\": \"What is fee structure?\"}'"
        },
        "test_form": '<form method="POST" action="/chat"><input name="question" placeholder="Ask a question..." style="width:300px;padding:5px;"><button type="submit">Ask</button></form>'
    }


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """
    Get RAG service statistics.

    Returns:
    - embedding_model_loaded: Whether the embedding model is loaded
    - cached_embeddings: Number of cached embeddings
    - documents: Number of documents indexed
    - data_path: Path to the data folder
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please check server logs.",
        )

    try:
        stats_dict = rag_service.get_rag_stats()
        return StatsResponse(
            embedding_model_loaded=stats_dict.get("embedding_model_loaded", False),
            cached_embeddings=stats_dict.get("cached_embeddings", 0),
            documents=stats_dict.get("documents", 0),
            data_path=stats_dict.get("data_path", "unknown"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}",
        )


# -----------------------------------------------------------------------------
# Main entry point (for direct execution)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"\nStarting FastAPI server on {host}:{port}")
    print("API docs available at: http://localhost:8000/docs")
    print()

    uvicorn.run(
        "api_app:app",
        host=host,
        port=port,
        reload=True,
    )
