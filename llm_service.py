#!/usr/bin/env python3
"""
LLM Service

Interface to language model backends: Ollama, llama.cpp, or OpenAI.
"""

import os
import requests


class LLMService:
    """Calls a local or remote LLM to generate answers."""

    def __init__(self, model_name: str, backend: str = "llamacpp"):
        self.model_name = model_name
        self.backend = backend
        self.base_url = self._get_base_url(backend)

    def _get_base_url(self, backend: str) -> str:
        """Return the base URL for the given backend."""
        urls = {
            "llamacpp": "http://localhost:8080",
            "ollama": "http://localhost:11434",
            "openai": "https://api.openai.com/v1",
        }
        url = urls.get(backend)
        if not url:
            raise ValueError(f"Unknown backend: {backend}")
        return url

    def generate_answer(self, prompt: str, question: str = "") -> str:
        """Generate an answer from the prompt using the configured backend."""
        try:
            if self.backend == "llamacpp":
                return self._llamacpp(prompt)
            if self.backend == "ollama":
                return self._ollama(prompt)
            if self.backend == "openai":
                return self._openai(prompt)
        except Exception as e:
            return f"LLM error: {e}"
        return "Unsupported backend"

    def _llamacpp(self, prompt: str) -> str:
        """Generate using a llama.cpp server."""
        response = requests.post(
            f"{self.base_url}/completion",
            json={
                "prompt": prompt,
                "n_predict": 256,
                "temperature": 0.7,
            },
            timeout=30,
        )
        if response.status_code == 200:
            return response.json().get("content", "").strip()
        return f"HTTP {response.status_code}"

    def _ollama(self, prompt: str) -> str:
        """Generate using Ollama with optimized parameters."""
        try:
            # Use optimized parameters for faster response
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.9,
                        "max_tokens": 100,  # Reduced from 150
                        "num_predict": 100,  # Reduced from 150
                        "repeat_penalty": 1.1,
                        "num_ctx": 2048,  # Context size
                        "num_batch": 512,  # Batch size
                        "num_thread": 4,  # Threads
                        "f16_kv": True,  # Use half precision
                        "use_mmap": True,  # Memory mapping
                        "use_mlock": False  # Don't lock memory
                    }
                },
                timeout=15,  # Reduced timeout
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                # Truncate if too long
                if len(result) > 500:
                    result = result[:500] + "..."
                return result
            else:
                return f"HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "LLM response timed out. Try a smaller question or check model performance."
        except Exception as e:
            return f"LLM error: {e}"

    def _openai(self, prompt: str) -> str:
        """Generate using OpenAI API."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OPENAI_API_KEY not set"

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
            },
            timeout=30,
        )
        if response.status_code != 200:
            return f"HTTP {response.status_code}"
        return response.json()["choices"][0]["message"]["content"].strip()


def create_llm_service(model_name: str, backend: str = "ollama") -> LLMService:
    """Create and return an LLMService instance with fallback options."""
    if backend == "mock":
        raise ValueError("Use a real backend: llamacpp, ollama, or openai")
    
    # Auto-detect best model if not specified
    if model_name == "auto":
        model_name = "gemma2:2b"  # Faster model by default
        backend = "ollama"
    
    return LLMService(model_name, backend)


def get_optimal_model() -> tuple[str, str]:
    """Get optimal model and backend for current system."""
    # Try faster models first
    fast_models = [
        ("qwen2:1.5b", "ollama"),      # User requested qwen2 - fastest
        ("qwen2:0.5b", "ollama"),      # Ultra-fast qwen2 variant
        ("qwen2.5:1.5b", "ollama"),   # Current fast model
        ("qwen2.5:3b", "ollama"),      # Original model
    ]
    
    for model, backend in fast_models:
        try:
            # Quick check if model is available
            response = requests.get(f"http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if model in models:
                    return model, backend
        except:
            pass
    
    # Fallback to default
    return "qwen2.5:3b", "ollama"
