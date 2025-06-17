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

