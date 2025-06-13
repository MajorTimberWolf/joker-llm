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
        
        # Groq-specific rate limiting (30 requests per minute = 2 seconds between calls)
        self.groq_rate_limit = self._is_groq_client()
        if self.groq_rate_limit:
            self.min_delay_between_calls = 2.1  # 2.1 seconds to stay under 30/min safely
            logger.info("Groq client detected - using conservative rate limiting (2.1s between calls)")
        
    def _is_groq_client(self) -> bool:
        """Check if the API client is a Groq client."""
        if not self.api_client:
            return False
        return hasattr(self.api_client, '_api_key') or 'groq' in str(type(self.api_client)).lower()
    
    def _select_optimal_groq_model(self) -> str:
        """Select the best Groq model based on expected API call volume."""
        # If we expect heavy usage (many API calls), prefer models with higher token limits
        if self.api_call_count > 20:  # Heavy workload detected
            # Use model with highest token/minute limit
            return "meta-llama/llama-4-scout-17b-16e-instruct"  # 30,000 tokens/min
        elif self.api_call_count > 10:  # Moderate workload
            return "gemma2-9b-it"  # 15,000 tokens/min
        else:  # Light workload, prioritize quality
            return "llama-3.3-70b-versatile"  # 12,000 tokens/min, best quality
    
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
                    # Mock response for development/testing
                    content = self._generate_mock_response(prompt)
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
    
    def _generate_mock_response(self, prompt: str) -> str:
        """
        Generate appropriate mock responses based on prompt content and topic.
        """
        prompt_lower = prompt.lower()
        
        # Extract topic from prompt if available
        topic = self._extract_topic_from_prompt(prompt)
        
        # Topic analysis mock
        if "analyze this topic for joke-writing potential" in prompt_lower:
            return self._generate_topic_analysis_mock(topic)
        
        # Basic angles mock
        elif "generate 5 distinct joke angles" in prompt_lower:
            return self._generate_basic_angles_mock(topic)
        
        # Advanced techniques mock
        elif "generate 3 sophisticated joke concepts" in prompt_lower:
            return self._generate_advanced_angles_mock(topic)
        
        # Combination/hybrid mock
        elif "create 3 hybrid joke concepts" in prompt_lower:
            return self._generate_hybrid_angles_mock(topic)
        
        # Joke creation mock - generate different jokes based on angle with randomization
        elif "create a joke using this approach" in prompt_lower:
            return self._generate_joke_mock(prompt_lower, topic)
        
        # Critique mock
        elif "analyze this joke for improvement" in prompt_lower:
            return """Critique: The joke structure is solid with good wordplay
Improvement Suggestions: Could strengthen the setup to emphasize the topic more"""
        
        # Refinement mock - improve jokes based on original content with randomization
        elif "improved version of the joke" in prompt_lower:
            return self._generate_refinement_mock(prompt_lower, topic)
        
        # Evaluation mocks with randomized scores
        elif "rate each joke on these 5 dimensions" in prompt_lower:
            # Generate randomized but reasonable scores
            cleverness = random.randint(5, 9)
            surprise = random.randint(4, 9)
            relatability = random.randint(6, 10)
            timing = random.randint(5, 9)
            overall = round((cleverness + surprise + relatability + timing) / 4, 1)
            
            return f"""JOKE: Sample joke being evaluated
CLEVERNESS: {cleverness}
SURPRISE: {surprise}
RELATABILITY: {relatability}
TIMING: {timing}
OVERALL: {overall}
REASONING: {random.choice(['Good wordplay and timing', 'Creative concept with solid execution', 'Relatable humor with clever twist', 'Strong setup and punchline', 'Original angle with good delivery'])}
---"""
        
        # Comparison mock with randomization
        elif "compare these two jokes" in prompt_lower:
            winner = random.choice(['A', 'B'])
            margin = random.choice(['Slightly', 'Moderately', 'Significantly'])
            reasoning = random.choice([
                'Better wordplay and more surprising punchline',
                'Stronger setup and clearer humor delivery',
                'More relatable concept with better timing',
                'More original approach and creative execution',
                'Clearer punchline and better comedic structure'
            ])
            return f"""WINNER: {winner}
REASONING: {reasoning}
MARGIN: {margin}"""
        
        # Simple scoring mock with randomization
        elif "rate how much each joke" in prompt_lower:
            score = random.randint(4, 10)
            reasoning = random.choice([
                'Clever wordplay makes it memorable',
                'Good timing and relatable humor',
                'Creative approach with solid execution',
                'Strong concept but could use refinement',
                'Original idea with effective delivery',
                'Well-structured joke with clear punchline'
            ])
            return f"""JOKE 1: {score}
REASONING: {reasoning}"""
        
        # Analysis mock
        elif "analyze the results of this joke generation" in prompt_lower:
            return f"""Analysis: The top jokes succeeded through clever wordplay combining {topic} terms with relatable experiences. The most effective approach was using misdirection and observational humor. This could be improved by exploring more diverse humor angles beyond just puns."""
        
        # Diversity mock
        elif "completely different in style and approach" in prompt_lower:
            return self._generate_diversity_mock(topic)
        
        # Default mock
        else:
            return f"Mock response for prompt type: {prompt[:50]}..."
    
    def _extract_topic_from_prompt(self, prompt: str) -> str:
        """Extract the topic from the prompt text."""
        # Try to find topic markers
        if "topic:" in prompt.lower():
            parts = prompt.lower().split("topic:")
            if len(parts) > 1:
                topic_part = parts[1].split('\n')[0].split('.')[0].strip(' "\'-')
                return topic_part
        
        # Check for common topic indicators
        if "artificial intelligence" in prompt.lower() or " ai " in prompt.lower():
            return "artificial intelligence"
        elif "bangalore" in prompt.lower() and "traffic" in prompt.lower():
            return "bangalore traffic"
        elif "traffic" in prompt.lower():
            return "traffic"
        elif "bangalore" in prompt.lower():
            return "bangalore"
        
        # Default fallback
        return "general topic"
    
    def _generate_topic_analysis_mock(self, topic: str) -> str:
        """Generate topic analysis based on the actual topic."""
        topic_lower = topic.lower()
        
        if "artificial intelligence" in topic_lower or "ai" in topic_lower:
            return """Context: This topic relates to technology and artificial intelligence systems.
Wordplay Opportunities: AI, artificial, intelligence, smart, learning, neural, deep
Cultural References: Sci-fi movies, tech industry, automation fears, robot jokes
Absurdity Potential: AI doing human activities, robots with emotions, machine consciousness"""
        
        elif "traffic" in topic_lower and "bangalore" in topic_lower:
            return """Context: This topic relates to urban transportation challenges in India's tech capital.
Wordplay Opportunities: jam, stuck, rush, hour, signal, lane, honk, auto, uber, metro
Cultural References: Bangalore IT culture, Indian traffic rules, auto-rickshaws, traffic police
Absurdity Potential: Extreme traffic situations, creative commuting solutions, road rage scenarios"""
        
        elif "traffic" in topic_lower:
            return """Context: This topic relates to transportation and commuting challenges.
Wordplay Opportunities: jam, stuck, rush, signal, lane, drive, park, speed, road
Cultural References: Road rage, commuting stress, parking problems, traffic laws
Absurdity Potential: Extreme traffic situations, alternative transport methods, driver behavior"""
        
        else:
            return f"""Context: This topic relates to {topic} and associated experiences.
Wordplay Opportunities: Various terms related to {topic}
Cultural References: Common experiences and stereotypes about {topic}
Absurdity Potential: Exaggerated situations and unexpected scenarios involving {topic}"""
    
    def _generate_basic_angles_mock(self, topic: str) -> str:
        """Generate basic joke angles based on the topic."""
        topic_lower = topic.lower()
        
        if "artificial intelligence" in topic_lower or "ai" in topic_lower:
            return """1. Pun approach: Play on AI terminology like "deep learning" or "neural networks"
2. Observational humor: AI trying to understand human behavior
3. Absurdist humor: AI doing mundane human tasks incorrectly
4. Character-based humor: Stereotypical robot personality traits
5. Situational irony: AI being bad at logical tasks"""
        
        elif "traffic" in topic_lower and "bangalore" in topic_lower:
            return """1. Pun approach: Play on traffic terms like "jam," "rush hour," "signal"
2. Observational humor: Bangalore commuter daily struggles
3. Absurdist humor: Extreme traffic situations and creative solutions
4. Character-based humor: Different types of Bangalore drivers
5. Situational irony: Tech capital with ancient traffic problems"""
        
        elif "traffic" in topic_lower:
            return """1. Pun approach: Play on traffic and driving terminology
2. Observational humor: Daily commuting experiences and frustrations
3. Absurdist humor: Extreme traffic scenarios and road rage
4. Character-based humor: Different types of drivers and their quirks
5. Situational irony: Modern life vs. ancient traffic systems"""
        
        else:
            return f"""1. Pun approach: Play on terminology related to {topic}
2. Observational humor: Common experiences with {topic}
3. Absurdist humor: Extreme scenarios involving {topic}
4. Character-based humor: Different types of people in {topic} situations
5. Situational irony: Modern expectations vs. {topic} reality"""
    
    def _generate_advanced_angles_mock(self, topic: str) -> str:
        """Generate advanced joke concepts based on the topic."""
        topic_lower = topic.lower()
        
        if "artificial intelligence" in topic_lower or "ai" in topic_lower:
            return """1. Misdirection: Set up AI as super smart, reveal it can't do simple task
2. Meta-humor: AI trying to understand why jokes are funny
3. Callback: Reference earlier AI training data in punchline"""
        
        elif "traffic" in topic_lower:
            return """1. Misdirection: Set up modern transport technology, reveal ancient traffic problems
2. Meta-humor: Traffic systems analyzing their own inefficiency
3. Callback: Reference to past traffic experiences in punchline"""
        
        else:
            return f"""1. Misdirection: Set up expectations about {topic}, reveal unexpected reality
2. Meta-humor: Self-awareness about {topic} situations
3. Callback: Reference earlier experiences with {topic}"""
    
    def _generate_hybrid_angles_mock(self, topic: str) -> str:
        """Generate hybrid joke concepts based on the topic."""
        topic_lower = topic.lower()
        
        if "artificial intelligence" in topic_lower or "ai" in topic_lower:
            return """1. Hybrid: AI combined with cooking - smart fridge that judges your food choices
2. Hybrid: AI in medieval times - knight trying to use voice commands on sword
3. Hybrid: AI meets pets - robot dog that needs software updates instead of walks"""
        
        elif "traffic" in topic_lower and "bangalore" in topic_lower:
            return """1. Hybrid: Bangalore traffic meets dating - using traffic apps to find romantic partners
2. Hybrid: Traffic signals as life coaches - giving relationship advice at red lights
3. Hybrid: Auto-rickshaw drivers as tech consultants - solving IT problems during rides"""
        
        elif "traffic" in topic_lower:
            return """1. Hybrid: Traffic meets cooking - using road terminology for recipes
2. Hybrid: Traffic signals as therapists - providing emotional guidance
3. Hybrid: Car horns as musical instruments - creating traffic symphonies"""
        
        else:
            return f"""1. Hybrid: {topic} meets cooking - using {topic} terms in kitchen scenarios
2. Hybrid: {topic} meets dating - applying {topic} concepts to relationships
3. Hybrid: {topic} meets technology - bringing modern tech to {topic} situations"""
    
    def _generate_joke_mock(self, prompt_lower: str, topic: str) -> str:
        """Generate joke content based on angle and topic."""
        topic_lower = topic.lower()
        
        if "traffic" in topic_lower and "bangalore" in topic_lower:
            jokes = [
                "Why do Bangalore techies love traffic jams? Because it's the only time their code compiles faster than they move!",
                "Bangalore traffic is so bad, Google Maps just shows 'Good luck!' instead of directions.",
                "In Bangalore, we don't say 'I'm stuck in traffic,' we say 'I'm participating in the world's longest parking lot.'",
                "Bangalore traffic moves so slowly, evolution happens faster - I saw a pedestrian grow a beard while crossing the street!",
                "Why don't Bangalore drivers use turn signals? Because by the time they change lanes, the destination has moved cities!"
            ]
            return f"""Outline: Setup about Bangalore traffic situation, punchline reveals absurd comparison
Joke: {random.choice(jokes)}"""
        
        elif "traffic" in topic_lower:
            jokes = [
                "My GPS got so frustrated with traffic, it started suggesting teleportation as the fastest route.",
                "Traffic was so bad today, I aged three years in the time it took to move one car length.",
                "Why do traffic lights change colors? Because they're embarrassed about how long they make people wait!",
                "I'm not stuck in traffic, I'm just part of the world's slowest parade!",
                "My car's fuel efficiency in traffic: 0.5 miles per gallon, 100% stress per minute."
            ]
            return f"""Outline: Setup about traffic frustration, punchline reveals absurd situation
Joke: {random.choice(jokes)}"""
        
        elif "artificial intelligence" in topic_lower or "ai" in topic_lower:
            jokes = [
                "Why did the AI go to therapy? Because it had too many deep learning issues!",
                "What did the neural network say at its job interview? 'I have deep experience and excellent pattern recognition!'",
                "Why did the machine learning model break up with its dataset? It said 'You're overfitting me!'"
            ]
            return f"""Outline: Setup about AI being advanced, punchline about technical terms
Joke: {random.choice(jokes)}"""
        
        else:
            return f"""Outline: Setup about {topic} situation, punchline reveals unexpected twist
Joke: Why did the {topic} expert become a comedian? Because they realized {topic} was already a joke!"""
    
    def _generate_refinement_mock(self, prompt_lower: str, topic: str) -> str:
        """Generate refined jokes based on original content and topic."""
        topic_lower = topic.lower()
        
        if "traffic" in topic_lower and "bangalore" in topic_lower:
            refined_jokes = [
                "Bangalore traffic is so legendary, NASA uses it to test the patience of astronauts - if you can handle Outer Ring Road during rush hour, you're ready for a mission to Mars!",
                "In Bangalore, we measure distance in time zones - 'Sir, your office is only 5 kilometers away.' 'Yes, but that's 3 hours in Bangalore Standard Time!'",
                "Bangalore traffic moves so slowly, archaeologists have found fossils of commuters who started their journey in 2019 and are still traveling to Electronic City!",
                "Why do Bangalore IT companies have so many floors? Because by the time employees reach office through traffic, they need elevators just to remember what floor they work on!"
            ]
            return f"""Improved Joke: {random.choice(refined_jokes)}"""
        
        elif "traffic" in topic_lower:
            refined_jokes = [
                "My commute is so long, my car's GPS started charging me rent for living in it during rush hour!",
                "Traffic was so bad today, I saw a snail pass my car and give me a sympathetic look.",
                "I don't need a gym membership - pushing my car through traffic for 3 hours a day is cardio enough!",
                "My driving instructor said 'Always keep a safe distance.' In this traffic, that distance is measured in geological eras."
            ]
            return f"""Improved Joke: {random.choice(refined_jokes)}"""
        
        else:
            refined_jokes = [
                f"The {topic} situation got so extreme, even the experts started googling 'how to escape {topic}.'",
                f"Why did the {topic} enthusiast become a stand-up comedian? Because {topic} was already providing all the material!",
                f"I'm not saying {topic} is complicated, but even quantum physicists look at it and say 'That's too confusing!'"
            ]
            return f"""Improved Joke: {random.choice(refined_jokes)}"""
    
    def _generate_diversity_mock(self, topic: str) -> str:
        """Generate diverse jokes based on topic."""
        topic_lower = topic.lower()
        
        if "traffic" in topic_lower and "bangalore" in topic_lower:
            return """New Joke 1: Why did the Bangalore commuter become a philosopher? Because spending 4 hours in traffic every day gives you plenty of time to contemplate the meaning of life!
New Joke 2: Bangalore traffic is the world's largest meditation retreat - you sit still for hours, practice patience, and achieve enlightenment about alternative career paths.
New Joke 3: In Bangalore, we don't have rush hour - we have 'slow hour,' 'slower hour,' and 'are-we-even-moving hour.'"""
        
        elif "traffic" in topic_lower:
            return """New Joke 1: Traffic is nature's way of teaching us that time is relative - 5 minutes can feel like eternity.
New Joke 2: I love traffic jams - where else can you watch three movies on your phone while traveling 2 blocks?
New Joke 3: My car has two speeds: parked and slightly less parked."""
        
        else:
            return f"""New Joke 1: {topic} walks into a bar. The bartender says "Why the long face?"
New Joke 2: What's the difference between {topic} and a magic trick? Magic tricks eventually end.
New Joke 3: I told my friend a joke about {topic}. They're still trying to figure out the punchline."""
    
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