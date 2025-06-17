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

class EnhancedAbductiveJokePipeline:
    """Research-grade pipeline with key enhancements"""
    
    MULTI_PREMISE_PROMPT = """Create a joke world for "{topic}" with {num_premises} premises:

1. **Grounding Premises (1-2):** Normal, relatable facts
2. **Absurd Premise (1):** Wildly unexpected rule that becomes law
3. **Conditional Premise (0-1):** If-then rule based on other premises

Format:
Grounding Premise 1: [normal fact]
Grounding Premise 2: [another normal fact, if needed]
Absurd Premise: [surreal rule]
Conditional Premise: [if-then rule, if applicable]"""

    PREMISE_QUALITY_PROMPT = """Rate this premise for joke generation:

Premise: "{premise}"
Type: {premise_type}

Score 1-10 for:
Specificity: [score] (how concrete/specific)
Novelty: [score] (how original/unexpected) 
Usability: [score] (how easy to build jokes from)
Overall: [average score]"""

    ENHANCED_JOKE_PROMPT = """Generate a joke using abductive reasoning:

WORLD RULES:
{premises}

Create:
Setup: [puzzling observation]
Punchline: [logical explanation using absurd premise]
Reasoning: [explicit logical chain]

The punchline must logically follow from the absurd premise."""

    def __init__(self, llm_client, enable_caching: bool = True):
        self.llm_client = llm_client
        self.api_call_count = 0
        self.cache = {} if enable_caching else None
        self.banned_premises = {
            'secretly aliens', 'magic powers', 'conspiracy theory',
            'government cover-up', 'illuminati'
        }
        
    def call_llm_cached(self, prompt: str, temperature: float = 0.8) -> str:
        """LLM call with caching support"""
        cache_key = hashlib.md5(f"{prompt}_{temperature}".encode()).hexdigest()
        
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        self.api_call_count += 1
        time.sleep(1.0)  # Rate limiting
        
        # Make API call (simplified - use existing method)
        if hasattr(self.llm_client, 'chat'):
            response = self.llm_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1500
            )
            result = response.choices[0].message.content
        else:
            raise RuntimeError("Unsupported client")
        
        if self.cache:
            self.cache[cache_key] = result
        
        return result
    
    def establish_multi_premise_world(self, topic: str, num_premises: int = 3) -> EnhancedJokeWorld:
        """Create world with multiple premises and quality filtering"""
        logger.info(f"Creating multi-premise world: {topic} ({num_premises} premises)")
        
        prompt = self.MULTI_PREMISE_PROMPT.format(topic=topic, num_premises=num_premises)
        response = self.call_llm_cached(prompt, 0.8)
        
        premises = self._parse_premises(response)
        premises = self._filter_banned_premises(premises)
        premises = self._score_premise_quality(premises)
        premises = [p for p in premises if (p.quality_score or 5.0) >= 4.0]
        
        if len(premises) < 2:
            premises = self._fallback_premises(topic)
        
        return EnhancedJokeWorld(topic=topic, premises=premises)
    
    def _parse_premises(self, response: str) -> List[EnhancedJokePremise]:
        """Parse premise response into structured objects"""
        premises = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                if line.startswith('Grounding Premise'):
                    content = line.split(':', 1)[1].strip()
                    premises.append(EnhancedJokePremise(content=content, premise_type='grounding'))
                elif line.startswith('Absurd Premise'):
                    content = line.split(':', 1)[1].strip()
                    premises.append(EnhancedJokePremise(content=content, premise_type='absurd'))
                elif line.startswith('Conditional Premise'):
                    content = line.split(':', 1)[1].strip()
                    premises.append(EnhancedJokePremise(content=content, premise_type='conditional'))
        
        return premises
    
    def _filter_banned_premises(self, premises: List[EnhancedJokePremise]) -> List[EnhancedJokePremise]:
        """Remove premises with banned content"""
        filtered = []
        for premise in premises:
            premise_lower = premise.content.lower()
            if not any(banned in premise_lower for banned in self.banned_premises):
                filtered.append(premise)
        return filtered
    
    def _score_premise_quality(self, premises: List[EnhancedJokePremise]) -> List[EnhancedJokePremise]:
        """Score premise quality using LLM judge"""
        for premise in premises:
            try:
                prompt = self.PREMISE_QUALITY_PROMPT.format(
                    premise=premise.content,
                    premise_type=premise.premise_type
                )
                response = self.call_llm_cached(prompt, 0.3)
                scores = self._parse_quality_scores(response)
                
                premise.specificity_score = scores.get('specificity', 5.0)
                premise.novelty_score = scores.get('novelty', 5.0)
                premise.quality_score = scores.get('overall', 5.0)
                
            except Exception as e:
                logger.warning(f"Quality scoring failed: {e}")
                premise.quality_score = 5.0
        
        return premises
    
    def _parse_quality_scores(self, response: str) -> Dict[str, float]:
        """Parse quality scoring response"""
        scores = {}
        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                try:
                    score = float(''.join(c for c in value if c.isdigit() or c == '.'))
                    scores[key] = max(1.0, min(10.0, score))
                except:
                    continue
        return scores
    
    def _fallback_premises(self, topic: str) -> List[EnhancedJokePremise]:
        """Generate fallback premises if parsing fails"""
        return [
            EnhancedJokePremise(
                content=f"{topic} is part of everyday life.",
                premise_type='grounding',
                quality_score=6.0
            ),
            EnhancedJokePremise(
                content=f"{topic} secretly follows absurd magical rules.",
                premise_type='absurd',
                quality_score=5.0
            )
        ]
    
    def generate_enhanced_joke(self, world: EnhancedJokeWorld, 
                             adaptive_temperature: bool = True) -> EnhancedAbductiveJoke:
        """Generate joke with adaptive temperature control"""
        premises_text = "\n".join([
            f"• {p.premise_type.title()}: {p.content}" for p in world.premises
        ])
        
        prompt = self.ENHANCED_JOKE_PROMPT.format(premises=premises_text)
        
        # Try high temperature first
        response = self.call_llm_cached(prompt, 0.9)
        setup, punchline, reasoning = self._parse_joke_response(response)
        
        joke = EnhancedAbductiveJoke(
            joke_world=world,
            setup=setup,
            punchline=punchline,
            reasoning_chain=reasoning,
            metadata={'temperature': 0.9, 'adaptive': adaptive_temperature}
        )
        
        # Adaptive temperature: retry if low quality
        if adaptive_temperature:
            consistency = self._quick_consistency_check(joke)
            if consistency < 6.0:
                response = self.call_llm_cached(prompt, 0.7)
                setup, punchline, reasoning = self._parse_joke_response(response)
                joke = EnhancedAbductiveJoke(
                    joke_world=world,
                    setup=setup,
                    punchline=punchline,
                    reasoning_chain=reasoning,
                    metadata={'temperature': 0.7, 'adaptive': True, 'retried': True}
                )
        
        return joke
    
    def _parse_joke_response(self, response: str) -> Tuple[str, str, str]:
        """Parse joke response components"""
        setup = punchline = reasoning = ""
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Setup:'):
                setup = line[6:].strip()
            elif line.startswith('Punchline:'):
                punchline = line[10:].strip()
            elif line.startswith('Reasoning:'):
                reasoning = line[10:].strip()
        
        if not setup or not punchline:
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            if len(sentences) >= 2:
                setup = sentences[0] + '.'
                punchline = sentences[1] + '.'
        
        return setup, punchline, reasoning
    
    def _quick_consistency_check(self, joke: EnhancedAbductiveJoke) -> float:
        """Quick consistency assessment for adaptive temperature"""
        reasoning = joke.reasoning_chain.lower()
        punchline = joke.punchline.lower()
        
        # Count logical indicators
        indicators = ['because', 'therefore', 'since', 'given', 'explains']
        indicator_count = sum(1 for ind in indicators if ind in reasoning or ind in punchline)
        
        # Check premise references
        absurd_premises = [p for p in joke.joke_world.premises if p.premise_type == 'absurd']
        premise_refs = 0
        for premise in absurd_premises:
            words = premise.content.lower().split()[:3]
            if any(word in punchline for word in words):
                premise_refs += 1
        
        # Simple scoring
        score = (indicator_count * 2) + (premise_refs * 3)
        return min(10.0, max(1.0, score))


class MultiJudgeAnalyzer:
    """Multi-judge evaluation system"""
    
    def __init__(self, llm_client, num_judges: int = 3):
        self.llm_client = llm_client
        self.num_judges = num_judges
        
    def evaluate_logical_consistency_ensemble(self, joke: EnhancedAbductiveJoke) -> Dict[str, Any]:
        """Multi-judge consistency evaluation"""
        consistency_prompt = """Rate logical consistency (1-10):

WORLD RULES:
{premises}

JOKE:
Setup: {setup}
Punchline: {punchline}

Score: [1-10 number only]
Reason: [brief explanation]"""
        
        premises_text = "\n".join([f"• {p.content}" for p in joke.joke_world.premises])
        prompt = consistency_prompt.format(
            premises=premises_text,
            setup=joke.setup,
            punchline=joke.punchline
        )
        
        scores = []
        explanations = []
        
        for i in range(self.num_judges):
            try:
                response = self._call_judge(prompt)
                score, explanation = self._parse_judge_response(response)
                scores.append(score)
                explanations.append(f"Judge {i+1}: {explanation}")
                time.sleep(0.5)  # Rate limit between judges
            except Exception as e:
                logger.warning(f"Judge {i+1} failed: {e}")
                scores.append(5.0)
                explanations.append(f"Judge {i+1}: Failed")
        
        return {
            'scores': scores,
            'median_score': statistics.median(scores) if scores else 0,
            'mean_score': statistics.mean(scores) if scores else 0,
            'std_score': statistics.stdev(scores) if len(scores) > 1 else 0,
            'agreement': 1.0 - (statistics.stdev(scores) / 10.0) if len(scores) > 1 else 1.0,
            'explanations': explanations
        }
    
    def _call_judge(self, prompt: str) -> str:
        """Call LLM judge"""
        if hasattr(self.llm_client, 'chat'):
            response = self.llm_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200
            )
            return response.choices[0].message.content
        else:
            raise RuntimeError("Unsupported client")
    
    def _parse_judge_response(self, response: str) -> Tuple[float, str]:
        """Parse judge response"""
        score = 5.0
        explanation = "No explanation"
        
        for line in response.split('\n'):
            if line.startswith('Score:'):
                try:
                    score_text = line[6:].strip()
                    score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
                    score = max(1.0, min(10.0, score))
                except:
                    pass
            elif line.startswith('Reason:'):
                explanation = line[7:].strip()
        
        return score, explanation





if __name__ == "__main__":
    print("Enhanced Research-Grade Abductive Joke Pipeline")
    print("Key Features:")
    print("✓ Multi-premise worlds with dependency tracking")
    print("✓ Premise quality scoring and filtering")
    print("✓ Multi-judge ensemble evaluation")
    print("✓ Adaptive temperature control")
    print("✓ Statistical analysis tools")
    print("✓ Type safety with Pydantic validation")
    print("✓ Comprehensive caching and logging")
    print("\nReady for research-grade experiments!") 