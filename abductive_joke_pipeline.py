#!/usr/bin/env python3
"""
Abductive Joke Pipeline
An implementation of joke generation using formal reasoning principles.

Instead of traditional brainstorming, this system creates jokes by establishing 
logical "worlds" with specific premises and then using abductive reasoning to 
generate surprising but internally consistent punchlines.
"""

import json
import time
import random
import logging
import statistics
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from joke_plan_search_core import JokePlanSearch, BiasConfig, JokeCandidate

logger = logging.getLogger(__name__)

@dataclass
class JokePremise:
    """Represents a single premise in the joke world"""
    def __init__(self, content: str, premise_type: str):
        self.content = content
        self.premise_type = premise_type  # "grounding" or "absurd"
    
    def to_dict(self) -> dict:
        return {
            'content': self.content,
            'premise_type': self.premise_type
        }

@dataclass
class JokeWorld:
    """Represents the complete world context for a joke"""
    def __init__(self, topic: str, grounding_premise: JokePremise, absurd_premise: JokePremise):
        self.topic = topic
        self.grounding_premise = grounding_premise
        self.absurd_premise = absurd_premise
    
    def to_dict(self) -> dict:
        return {
            'topic': self.topic,
            'grounding_premise': self.grounding_premise.to_dict(),
            'absurd_premise': self.absurd_premise.to_dict()
        }

@dataclass
class AbductiveJoke:
    """Represents a complete joke generated through abductive reasoning"""
    def __init__(self, joke_world: JokeWorld, setup: str, punchline: str, 
                 reasoning_chain: str = "", metadata: Dict[str, Any] = None):
        self.joke_world = joke_world
        self.setup = setup
        self.punchline = punchline
        self.reasoning_chain = reasoning_chain
        self.metadata = metadata or {}
        self.scores = {}
        self.evaluation_notes = ""
    
    def get_full_joke(self) -> str:
        """Returns the complete joke as a single string"""
        return f"{self.setup} {self.punchline}".strip()
    
    def to_dict(self) -> dict:
        return {
            'joke_world': self.joke_world.to_dict(),
            'setup': self.setup,
            'punchline': self.punchline,
            'reasoning_chain': self.reasoning_chain,
            'metadata': self.metadata,
            'scores': self.scores,
            'evaluation_notes': self.evaluation_notes
        }

class AbductiveJokePipeline:
    """Main pipeline for generating jokes using abductive reasoning"""
    
    # Exact prompt structures as specified
    PREMISE_GENERATION_PROMPT = """You are an expert in comedy and logic. For the given topic, your task is to establish a "joke world" by defining a set of premises or rules. You must provide two types of premises:

1. **Grounding Premise:** A true, normal, or stereotypical fact about the topic. This makes the joke relatable and establishes shared understanding.

2. **Absurd Premise:** A completely unexpected or surreal rule that will be treated as true within this specific joke's world. This should be creative, specific, and the primary source of humor potential.

Topic: {topic}

Requirements:
- The grounding premise should be something most people would agree is true or typical
- The absurd premise should be wildly unexpected but specific enough to build jokes around
- Both premises should be stated as facts within this joke's universe
- Avoid generic absurdity - make it memorably specific

Provide your response in this exact format:
Grounding Premise: [A normal, factual premise about the topic.]
Absurd Premise: [A strange, surreal, or hilariously specific rule you are inventing for this world.]

Examples for reference:
Topic: Libraries
Grounding Premise: Libraries are quiet places where people go to read and study.
Absurd Premise: All librarians are actually retired secret agents who use the Dewey Decimal System as a complex code for their former operations.

Topic: Coffee Shops  
Grounding Premise: Coffee shops serve caffeinated beverages and often have WiFi for customers.
Absurd Premise: Every coffee order is actually a spell, and baristas are low-level wizards who get increasingly powerful based on drink complexity."""

    ABDUCTIVE_JOKE_PROMPT = """Your task is to create a complete joke based on a pre-defined "world" using abductive reasoning principles. The joke must logically follow the rules of this world, treating even absurd premises as absolute truth.

JOKE WORLD CONTEXT:
- Topic: {topic}
- Grounding Premise: {grounding_premise}
- Absurd Premise: {absurd_premise}

ABDUCTIVE REASONING STRUCTURE:
Your joke should follow this logical pattern:
1. Setup: Present a situation or observation that seems puzzling or noteworthy
2. Implicit Question: The setup should make the audience wonder "why?" or "how?"
3. Punchline: Provide a surprising explanation that makes perfect sense ONLY because of the absurd premise

REQUIREMENTS:
- The setup must acknowledge or build on the grounding premise (relatability)
- The punchline must derive its logic from the absurd premise
- The explanation should be the "least expected but most logical" given the world rules
- Avoid obvious or predictable connections
- The joke should feel like a genuine insight within this absurd world

Format your response exactly as:
Setup: [Create a situation, question, or observation that needs explanation]
Punchline: [Provide the surprising but internally logical explanation]

Example Pattern:
World: Grounding = "Cats are independent pets", Absurd = "Cats secretly run a parallel economy using hairballs as currency"
Setup: "I wondered why my cat was so upset about the new vacuum cleaner."
Punchline: "Turns out I've been destroying the Federal Reserve of cat money every week."
"""

    def __init__(self, llm_client, bias_config: BiasConfig = None):
        """Initialize the abductive joke pipeline"""
        self.llm_client = llm_client
        self.bias_config = bias_config or BiasConfig()
        self.api_call_count = 0
        self.last_api_call = 0
        self.min_delay_between_calls = 1.0
        
        # Cache for premise generation to avoid redundant calls
        self.premise_cache = {}
        
        # Experimental tracking
        self.experiment_results = {}
        self.generated_jokes = []
        
    def call_llm(self, prompt: str, temperature: float = None) -> str:
        """Make an API call with rate limiting"""
        if temperature is None:
            temperature = self.bias_config.generation_temperature
            
        # Rate limiting
        time_since_last_call = time.time() - self.last_api_call
        if time_since_last_call < self.min_delay_between_calls:
            time.sleep(self.min_delay_between_calls - time_since_last_call)
        
        self.last_api_call = time.time()
        self.api_call_count += 1
        
        # Use the same LLM calling pattern as the existing pipeline
        if hasattr(self.llm_client, 'chat'):
            if hasattr(self.llm_client, '_api_key') or 'groq' in str(type(self.llm_client)).lower():
                # Groq API client
                response = self.llm_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            else:
                # OpenAI-style client
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1000
                )
                return response.choices[0].message.content
        elif hasattr(self.llm_client, 'messages'):
            # Anthropic-style client
            response = self.llm_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            raise RuntimeError(f"Unsupported LLM client type: {type(self.llm_client)}")
    
    def establish_joke_world_premises(self, topic: str) -> JokeWorld:
        """Generate grounding and absurd premises for a topic"""
        logger.info(f"Establishing joke world premises for topic: {topic}")
        
        # Check cache first
        if topic in self.premise_cache:
            logger.info("Using cached premises")
            return self.premise_cache[topic]
        
        prompt = self.PREMISE_GENERATION_PROMPT.format(topic=topic)
        response = self.call_llm(prompt, temperature=0.8)
        
        # Parse the response
        grounding_premise = None
        absurd_premise = None
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Grounding Premise:'):
                grounding_premise = JokePremise(
                    content=line[18:].strip(),
                    premise_type="grounding"
                )
            elif line.startswith('Absurd Premise:'):
                absurd_premise = JokePremise(
                    content=line[15:].strip(), 
                    premise_type="absurd"
                )
        
        if not grounding_premise or not absurd_premise:
            # Fallback parsing if format isn't exactly followed
            grounding_premise, absurd_premise = self._parse_premise_fallback(response, topic)
            if not grounding_premise or not absurd_premise:
                raise ValueError(f"Failed to parse premises from response: {response}")
        
        joke_world = JokeWorld(topic, grounding_premise, absurd_premise)
        
        # Cache the result
        self.premise_cache[topic] = joke_world
        
        logger.info(f"Generated premises - Grounding: {grounding_premise.content[:50]}...")
        logger.info(f"Generated premises - Absurd: {absurd_premise.content[:50]}...")
        
        return joke_world
    
    def _parse_premise_fallback(self, response: str, topic: str) -> Tuple[JokePremise, JokePremise]:
        """Fallback parsing method if structured format fails"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Look for any lines that might be premises
        potential_premises = []
        for line in lines:
            if len(line) > 20 and not line.startswith('Topic:') and not line.startswith('Requirements:'):
                potential_premises.append(line)
        
        if len(potential_premises) >= 2:
            grounding = JokePremise(potential_premises[0], "grounding")
            absurd = JokePremise(potential_premises[1], "absurd")
            return grounding, absurd
        
        # Ultimate fallback - generate basic premises
        grounding = JokePremise(f"{topic} is a common part of everyday life.", "grounding")
        absurd = JokePremise(f"{topic} secretly operates according to bizarre magical rules.", "absurd")
        return grounding, absurd
    
