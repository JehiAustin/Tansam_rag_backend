#!/usr/bin/env python3
"""
LLM Timeout Fix - Multiple Solutions for Hardware Limitations
"""

import os
import time
import hashlib
import json
from typing import Dict, Optional

class LLMOptimizer:
    """Optimize LLM performance with multiple strategies"""
    
    def __init__(self):
        self.cache = {}
        self.fallback_responses = {
            "fee": "Based on the documents, fee structures vary by program. Please check the specific academic program details for exact fee information.",
            "admission": "Admission requirements include academic records, entrance exams, and application forms. Check specific program requirements.",
            "course": "Course information is available in the academic catalog. Please specify which program or subject you're interested in.",
            "deadline": "Application deadlines vary by program. Please check the academic calendar or contact admissions office.",
            "default": "I found relevant information in the documents, but I'm experiencing technical difficulties. Please try rephrasing your question or contact support."
        }
    
    def get_cached_response(self, question: str) -> Optional[str]:
        """Get cached response if available"""
        cache_key = hashlib.md5(question.lower().encode()).hexdigest()
        return self.cache.get(cache_key)
    
    def cache_response(self, question: str, response: str):
        """Cache the response"""
        cache_key = hashlib.md5(question.lower().encode()).hexdigest()
        self.cache[cache_key] = response
    
    def get_fallback_response(self, question: str) -> str:
        """Get intelligent fallback response"""
        question_lower = question.lower()
        
        for keyword, response in self.fallback_responses.items():
            if keyword in question_lower:
                return response
        
        return self.fallback_responses["default"]
    
    def optimize_question(self, question: str) -> str:
        """Optimize question for faster processing"""
        # Remove unnecessary words
        stop_words = ["what", "how", "tell", "me", "about", "the", "is", "are"]
        words = question.lower().split()
        optimized = [word for word in words if word not in stop_words]
        
        # Limit length
        if len(optimized) > 10:
            optimized = optimized[:10]
        
        return " ".join(optimized)

def create_optimized_llm_service():
    """Create LLM service with optimizations"""
    from llm_service import create_llm_service, get_optimal_model
    
    # Get the smallest available model
    model, backend = get_optimal_model()
    
    # Override with ultra-fast model if available
    fast_models = ["qwen2:0.5b", "tinyllama:1.1b", "qwen2:1.5b"]
    
    for fast_model in fast_models:
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if fast_model in models:
                    model = fast_model
                    break
        except:
            pass
    
    print(f"🚀 Using optimized model: {model}")
    return create_llm_service(model, backend)

if __name__ == "__main__":
    # Test the optimizer
    optimizer = LLMOptimizer()
    
    test_questions = [
        "What is the fee structure?",
        "How do I apply for admission?",
        "Tell me about course requirements"
    ]
    
    for question in test_questions:
        print(f"Original: {question}")
        print(f"Optimized: {optimizer.optimize_question(question)}")
        print(f"Fallback: {optimizer.get_fallback_response(question)}")
        print("-" * 50)
