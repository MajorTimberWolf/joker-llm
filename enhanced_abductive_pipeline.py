#!/usr/bin/env python3
"""
Enhanced Research-Grade Abductive Joke Pipeline
Implements key improvements from the research upgrade plan.
"""

import json
import time
import random
import logging
import statistics
import hashlib
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
import numpy as np
from abductive_joke_pipeline import AbductiveJokePipeline, JokeWorld, AbductiveJoke, JokePremise

logger = logging.getLogger(__name__)

# Enhanced Models with Type Safety
class EnhancedJokePremise(BaseModel):
    """Enhanced premise with quality scoring and dependencies"""
    content: str = Field(..., min_length=10)
    premise_type: str = Field(..., regex="^(grounding|absurd|conditional)$")
    quality_score: Optional[float] = Field(None, ge=1.0, le=10.0)
    specificity_score: Optional[float] = Field(None, ge=1.0, le=10.0)
    novelty_score: Optional[float] = Field(None, ge=1.0, le=10.0)
    dependencies: List[str] = Field(default_factory=list)
    premise_id: str = Field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])

class EnhancedJokeWorld(BaseModel):
    """Enhanced world with multiple premises and dependency tracking"""
    topic: str = Field(..., min_length=1)
    premises: List[EnhancedJokePremise] = Field(..., min_items=2)
    premise_graph: Dict[str, List[str]] = Field(default_factory=dict)
    world_id: str = Field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    
    @validator('premises')
    def validate_premises(cls, v):
        types = [p.premise_type for p in v]
        if 'grounding' not in types or 'absurd' not in types:
            raise ValueError('Must have at least one grounding and one absurd premise')
        return v

class EnhancedAbductiveJoke(BaseModel):
    """Enhanced joke with comprehensive metadata"""
    joke_world: EnhancedJokeWorld
    setup: str = Field(..., min_length=5)
    punchline: str = Field(..., min_length=5)
    reasoning_chain: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    scores: Dict[str, float] = Field(default_factory=dict)
    similarity_hash: Optional[str] = Field(None)
    
    def __init__(self, **data):
        super().__init__(**data)
        content = f"{self.setup} {self.punchline}".lower()
        self.similarity_hash = hashlib.md5(content.encode()).hexdigest()

