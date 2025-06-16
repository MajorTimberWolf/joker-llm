"""
JokePlanSearch: Core Implementation
A comprehensive LLM-based joke generation and evaluation system following the PlanSearch methodology.
"""

import json
import time
import random
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class JokeCandidate:
    """Data structure for storing joke generation and evaluation data."""
    angle: str
    outline: str
    full_joke: str
    refined_joke: str = ""
    critique: str = ""
    improvement_suggestions: str = ""
    scores: Dict[str, float] = None
    comparative_wins: int = 0
    comparative_losses: int = 0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self):
        if self.scores is None:
            self.scores = {}

@dataclass
class TopicAnalysis:
    """Data structure for storing topic analysis results."""
    original_topic: str
    context_analysis: str
    wordplay_potential: str
    cultural_references: str
    absurdity_potential: str

class BiasConfig:
    """Configuration for bias mitigation techniques."""
    def __init__(self):
        self.evaluation_rounds = 3
        self.judge_temperature = 0.3
        self.generation_temperature = 0.7
        self.shuffle_iterations = 2
        self.min_comparisons_per_joke = 3
        self.confidence_threshold = 0.8

class JokePlanSearch:
    """
    Main class implementing the JokePlanSearch methodology for computational humor.
    """
    
    def __init__(self, api_client=None, bias_config: BiasConfig = None):
        """
        Initialize JokePlanSearch with a REQUIRED real Groq API client.
        If ``api_client`` is ``None`` the constructor will attempt to create a
        Groq client from the ``GROQ_API_KEY`` environment variable.  Any
        failure to obtain a real client raises immediately so that the system
        never falls back to mock data.
        """

        # ------------------------------------------------------------------
        # Fail-fast: ensure we have a usable API client
        # ------------------------------------------------------------------
        if api_client is None:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    import groq  # Deferred import so package is optional until needed
                    api_client = groq.Groq(api_key=groq_key)
                except ImportError as exc:
                    raise ImportError(
                        "Groq SDK is not installed. Install with `pip install groq`."
                    ) from exc
            else:
                raise ValueError(
                    "No API client supplied and GROQ_API_KEY is not set. "
                    "Configure a real Groq client before instantiating JokePlanSearch."
                )

        self.api_client = api_client
        self.bias_config = bias_config or BiasConfig()
        self.topic_analysis: Optional[TopicAnalysis] = None
        self.joke_angles: List[str] = []
        self.joke_candidates: List[JokeCandidate] = []
        self.evaluation_results: Dict[str, Any] = {}
        self.api_call_count = 0
        self.total_tokens_used = 0
        
        # Rate limiting
        self.last_api_call = 0
        self.min_delay_between_calls = 1.0  # seconds
        
        # Groq-specific rate limiting - Use optimized settings for different models
        self.groq_rate_limit = self._is_groq_client()
        if self.groq_rate_limit:
            # Check if we can use the faster Qwen model (60 requests/min)
            self.preferred_model = self._select_optimal_groq_model()
            if "qwen" in self.preferred_model.lower():
                self.min_delay_between_calls = 1.1  # 60 requests/min = 1 second, use 1.1 for safety
                logger.info("Qwen model detected - using optimized rate limiting (1.1s between calls, 60 req/min)")
            else:
                self.min_delay_between_calls = 2.1  # 30 requests/min = 2 seconds, use 2.1 for safety
                logger.info("Groq client detected - using conservative rate limiting (2.1s between calls, 30 req/min)")
        
    def _is_groq_client(self) -> bool:
        """Check if the API client is a Groq client."""
        if not self.api_client:
            return False
        return hasattr(self.api_client, '_api_key') or 'groq' in str(type(self.api_client)).lower()
    
    def _select_optimal_groq_model(self) -> str:
        """Select the best Groq model based on expected API call volume and free tier optimization."""
        # For free tier users, prioritize Qwen model (60 requests/min vs 30 for others)
        expected_calls = getattr(self, 'expected_api_calls', 30)  # Default estimate
        
        # Strategy:
        # • Heavy > 45 calls → pick Qwen for 60 req/min.
        # • Medium 31-45 calls → still Qwen.
        # • Light ≤ 30 calls (typical free-tier run) → use the new Scout 17B model.
        if expected_calls > 45:  # Very high workload
            return "qwen/qwen3-32b"
        elif expected_calls > 30:  # High but within one-minute budget
            return "qwen/qwen3-32b"
        else:  # Free-tier and general low/medium workloads
            return "meta-llama/llama-4-scout-17b-16e-instruct"
    
    def call_llm(self, prompt: str, temperature: float = None, max_retries: int = 3) -> str:
        """
        Make an API call to the LLM with error handling and rate limiting.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Temperature for response generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            The LLM's response as a string
        """
        if temperature is None:
            temperature = self.bias_config.generation_temperature
            
        # Rate limiting
        time_since_last_call = time.time() - self.last_api_call
        if time_since_last_call < self.min_delay_between_calls:
            time.sleep(self.min_delay_between_calls - time_since_last_call)
        
        for attempt in range(max_retries):
            try:
                self.last_api_call = time.time()
                self.api_call_count += 1
                
                # Adapt this section based on your chosen API client
                if hasattr(self.api_client, 'chat'):
                    # Check if it's a Groq client (has api_key attribute set in a specific way)
                    if hasattr(self.api_client, '_api_key') or 'groq' in str(type(self.api_client)).lower():
                        # Groq API client (OpenAI-compatible interface)
                        # Use the best available model based on rate limits
                        groq_model = self._select_optimal_groq_model()
                        response = self.api_client.chat.completions.create(
                            model=groq_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=1000
                        )
                        content = response.choices[0].message.content
                        if hasattr(response, 'usage') and response.usage:
                            self.total_tokens_used += response.usage.total_tokens
                    else:
                        # OpenAI-style client
                        response = self.api_client.chat.completions.create(
                            model="gpt-4",  # Adjust model as needed
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=1000
                        )
                        content = response.choices[0].message.content
                        self.total_tokens_used += response.usage.total_tokens
                    
                elif hasattr(self.api_client, 'messages'):  # Anthropic-style client
                    response = self.api_client.messages.create(
                        model="claude-3-sonnet-20240229",  # Adjust model as needed
                        max_tokens=1000,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.content[0].text
                    self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens
                    
                else:
                    raise RuntimeError(
                        "Unsupported or missing API client. Provide a valid Groq client; "
                        "mock responses have been removed from the system."
                    )
                
                logger.info(f"API call {self.api_call_count} successful")
                return content.strip()
                
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All API call attempts failed for prompt: {prompt[:100]}...")
                    raise
                    
    def call_llm_judge(self, prompt: str) -> str:
        """
        Make an API call optimized for evaluation tasks (lower temperature).
        """
        return self.call_llm(prompt, temperature=self.bias_config.judge_temperature)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance and usage metrics for the session."""
        return {
            "api_calls_made": self.api_call_count,
            "total_tokens_used": self.total_tokens_used,
            "jokes_generated": len(self.joke_candidates),
            "angles_explored": len(self.joke_angles),
            "successful_evaluations": len([c for c in self.joke_candidates if c.scores])
        }

def setup_logging_for_colab():
    """Configure logging for Google Colab environment."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    ) 