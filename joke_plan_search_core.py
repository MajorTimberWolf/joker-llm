"""
JokePlanSearch: Core Implementation
A comprehensive LLM-based joke generation and evaluation system following the PlanSearch methodology.
"""

import json
import time
import random
import logging
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
        Initialize JokePlanSearch with API client and configuration.
        
        Args:
            api_client: Initialized LLM API client (OpenAI, Anthropic, Groq, etc.)
            bias_config: Configuration for bias mitigation techniques
        """
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
                if hasattr(self.api_client, 'chat'):  # OpenAI-style client
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
                    # Mock response for development/testing
                    content = f"Mock response for: {prompt[:50]}..."
                    logger.warning("Using mock response - implement your API client integration")
                
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

# Utility functions
def create_mock_api_client():
    """
    Create a mock API client for testing purposes.
    Replace this with your actual API client initialization.
    """
    class MockClient:
        def __init__(self):
            self.chat = MockChat()
            
    class MockChat:
        def __init__(self):
            self.completions = MockCompletions()
            
    class MockCompletions:
        def create(self, **kwargs):
            class MockResponse:
                def __init__(self):
                    self.choices = [MockChoice()]
                    self.usage = MockUsage()
                    
            class MockChoice:
                def __init__(self):
                    self.message = MockMessage()
                    
            class MockMessage:
                def __init__(self):
                    self.content = "This is a mock joke response."
                    
            class MockUsage:
                def __init__(self):
                    self.total_tokens = 100
                    
            return MockResponse()
    
    return MockClient()

def setup_logging_for_colab():
    """Configure logging for Google Colab environment."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    ) 