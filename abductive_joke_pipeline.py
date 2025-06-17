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

