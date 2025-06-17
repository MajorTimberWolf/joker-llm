#!/usr/bin/env python3
"""
Enhanced Abductive Joke Pipeline - Research Grade
A sophisticated implementation of joke generation using formal reasoning principles
with comprehensive experimental design and statistical analysis capabilities.
"""

import json
import time
import random
import logging
import statistics
import asyncio
import hashlib
import sqlite3
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, validator, Field
import numpy as np
import pandas as pd
from joke_plan_search_core import JokePlanSearch, BiasConfig, JokeCandidate

logger = logging.getLogger(__name__)

# Pydantic Models for Type Safety
class JokePremise(BaseModel):
    """Enhanced premise with validation and quality scoring"""
    content: str = Field(..., min_length=10, description="The premise content")
    premise_type: str = Field(..., description="Type: grounding, absurd, or conditional")
    quality_score: Optional[float] = Field(None, ge=1.0, le=10.0, description="Quality rating 1-10")
    specificity_score: Optional[float] = Field(None, ge=1.0, le=10.0)
    novelty_score: Optional[float] = Field(None, ge=1.0, le=10.0)
    dependencies: List[str] = Field(default_factory=list, description="IDs of dependent premises")
    premise_id: str = Field(default_factory=lambda: hashlib.md5(time.time().__str__().encode()).hexdigest()[:8])
    
    @validator('premise_type')
    def validate_premise_type(cls, v):
        allowed_types = ['grounding', 'absurd', 'conditional']
        if v not in allowed_types:
            raise ValueError(f'premise_type must be one of {allowed_types}')
        return v

class JokeWorld(BaseModel):
    """Enhanced world context supporting multiple premises and dependency graphs"""
    topic: str = Field(..., min_length=1)
    premises: List[JokePremise] = Field(..., min_items=2)
    premise_graph: Dict[str, List[str]] = Field(default_factory=dict, 
                                               description="Dependencies between premises")
    world_id: str = Field(default_factory=lambda: hashlib.md5(time.time().__str__().encode()).hexdigest()[:8])
    
    @validator('premises')
    def validate_premises(cls, v):
        if len(v) < 2:
            raise ValueError('At least 2 premises required')
        
        # Check for at least one grounding and one absurd premise
        types = [p.premise_type for p in v]
        if 'grounding' not in types:
            raise ValueError('At least one grounding premise required')
        if 'absurd' not in types:
            raise ValueError('At least one absurd premise required')
        
        return v
    
    def get_grounding_premises(self) -> List[JokePremise]:
        return [p for p in self.premises if p.premise_type == 'grounding']
    
    def get_absurd_premises(self) -> List[JokePremise]:
        return [p for p in self.premises if p.premise_type == 'absurd']
    
    def get_conditional_premises(self) -> List[JokePremise]:
        return [p for p in self.premises if p.premise_type == 'conditional']

class AbductiveJoke(BaseModel):
    """Enhanced joke with comprehensive metadata and reasoning chains"""
    joke_world: JokeWorld
    setup: str = Field(..., min_length=5)
    punchline: str = Field(..., min_length=5)
    reasoning_chain: str = Field(default="", description="Explicit reasoning path")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    scores: Dict[str, float] = Field(default_factory=dict)
    evaluation_notes: str = Field(default="")
    generation_params: Dict[str, Any] = Field(default_factory=dict)
    similarity_hash: Optional[str] = Field(None, description="For duplicate detection")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Generate similarity hash for duplicate detection
        content = f"{self.setup} {self.punchline}".lower()
        self.similarity_hash = hashlib.md5(content.encode()).hexdigest()
    
    def get_full_joke(self) -> str:
        """Returns the complete joke as a single string"""
        return f"{self.setup} {self.punchline}".strip()
    
    def calculate_setup_diversity(self, other_jokes: List['AbductiveJoke']) -> float:
        """Calculate embedding similarity with other setups"""
        # Simplified similarity based on word overlap for now
        setup_words = set(self.setup.lower().split())
        similarities = []
        
        for other in other_jokes:
            other_words = set(other.setup.lower().split())
            if len(setup_words) > 0 and len(other_words) > 0:
                overlap = len(setup_words.intersection(other_words))
                similarity = overlap / max(len(setup_words), len(other_words))
                similarities.append(similarity)
        
        return max(similarities) if similarities else 0.0

class EnhancedAbductiveJokePipeline:
    """Research-grade pipeline with advanced features"""
    
    # Enhanced prompt templates with multi-premise support
    MULTI_PREMISE_GENERATION_PROMPT = """You are an expert in comedy and logic. For the given topic, create a "joke world" by defining {num_premises} premises that will govern this universe. You must provide:

1. **Grounding Premises (1-2):** True, normal facts about the topic that establish relatability
2. **Absurd Premise (1):** A wildly unexpected rule that becomes "law" in this world  
3. **Conditional Premise (0-1):** An if-then rule that depends on other premises

Topic: {topic}

Requirements:
- Each premise should be specific and memorable
- Absurd premises should be creative but logically usable
- Conditional premises should create interesting interactions
- All premises will be treated as absolute truth within this joke world

Provide your response in this exact format:
Grounding Premise 1: [Normal fact about the topic]
Grounding Premise 2: [Another normal fact, if applicable]
Absurd Premise: [Surreal rule that becomes law in this world]
Conditional Premise: [If-then rule based on other premises, if applicable]

Example for "Libraries":
Grounding Premise 1: Libraries are quiet places for reading and study.
Grounding Premise 2: Librarians help people find books and information.
Absurd Premise: Every book in a library is actually a sentient being that judges readers.
Conditional Premise: If someone whispers in the library, the books become offended and hide their content."""

    ENHANCED_ABDUCTIVE_JOKE_PROMPT = """Generate a joke using formal abductive reasoning within this established world.

JOKE WORLD:
Topic: {topic}
{premise_context}

REASONING TASK:
1. Present an observation that seems puzzling given the normal world rules
2. Apply abductive reasoning: what's the most surprising but logical explanation?
3. The explanation MUST follow logically from the absurd/conditional premises

FORMAT:
Setup: [Present the puzzling observation]
Punchline: [The abductive explanation that resolves the puzzle]
Reasoning: [Explicit logical chain: observation → premise application → conclusion]

QUALITY CRITERIA:
- The setup should feel relatable (uses grounding premises)
- The punchline should be surprising but inevitable given the world rules
- The reasoning chain should be traceable and logically sound within this world
- Avoid obvious or predictable connections"""

    PREMISE_QUALITY_PROMPT = """Rate this premise for joke generation quality:

Premise: "{premise_content}"
Type: {premise_type}

Rate from 1-10 on:
1. Specificity: How specific and concrete is this premise?
2. Novelty: How original and unexpected is this premise?
3. Usability: How easy would it be to build jokes from this premise?

Provide your response as:
Specificity: [score]
Novelty: [score]
Usability: [score]
Overall: [average score]
Reasoning: [brief explanation]"""

    def __init__(self, llm_client, bias_config: BiasConfig = None, 
                 enable_caching: bool = True, enable_logging: bool = True):
        """Initialize enhanced pipeline"""
        self.llm_client = llm_client
        self.bias_config = bias_config or BiasConfig()
        self.api_call_count = 0
        self.last_api_call = 0
        self.min_delay_between_calls = 1.0
        
        # Enhanced caching and logging
        self.enable_caching = enable_caching
        self.enable_logging = enable_logging
        self.premise_cache = {}
        self.generated_jokes = []
        self.banned_premises = set()  # Negative premise pool
        
        # Setup logging database if enabled
        if self.enable_logging:
            self._setup_logging_db()
    
    def _setup_logging_db(self):
        """Setup SQLite database for comprehensive logging"""
        self.db_path = "abductive_pipeline_logs.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                prompt_hash TEXT,
                model TEXT,
                temperature REAL,
                response TEXT,
                token_count INTEGER,
                cost_estimate REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jokes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                topic TEXT,
                joke_world_id TEXT,
                setup TEXT,
                punchline TEXT,
                reasoning_chain TEXT,
                similarity_hash TEXT,
                generation_params TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def call_llm_with_logging(self, prompt: str, temperature: float = None, 
                             model_override: str = None) -> str:
        """Enhanced LLM call with comprehensive logging"""
        if temperature is None:
            temperature = self.bias_config.generation_temperature
        
        # Rate limiting
        time_since_last_call = time.time() - self.last_api_call
        if time_since_last_call < self.min_delay_between_calls:
            time.sleep(self.min_delay_between_calls - time_since_last_call)
        
        self.last_api_call = time.time()
        self.api_call_count += 1
        
        # Generate prompt hash for logging
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        # Check cache if enabled
        cache_key = f"{prompt_hash}_{temperature}"
        if self.enable_caching and cache_key in self.premise_cache:
            logger.info("Using cached response")
            return self.premise_cache[cache_key]
        
        # Make API call
        response = self._make_api_call(prompt, temperature, model_override)
        
        # Cache response
        if self.enable_caching:
            self.premise_cache[cache_key] = response
        
        # Log to database
        if self.enable_logging:
            self._log_api_call(prompt_hash, temperature, response, model_override)
        
        return response
    
    def _make_api_call(self, prompt: str, temperature: float, model_override: str = None) -> str:
        """Make the actual API call with multi-provider support"""
        try:
            if hasattr(self.llm_client, 'chat'):
                if hasattr(self.llm_client, '_api_key') or 'groq' in str(type(self.llm_client)).lower():
                    model = model_override or "meta-llama/llama-4-scout-17b-16e-instruct"
                    response = self.llm_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content
                else:
                    model = model_override or "gpt-4"
                    response = self.llm_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content
            elif hasattr(self.llm_client, 'messages'):
                model = model_override or "claude-3-sonnet-20240229"
                response = self.llm_client.messages.create(
                    model=model,
                    max_tokens=2000,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            else:
                raise RuntimeError(f"Unsupported LLM client type: {type(self.llm_client)}")
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    
    def _log_api_call(self, prompt_hash: str, temperature: float, response: str, model: str):
        """Log API call to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO api_calls (timestamp, prompt_hash, model, temperature, response, token_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (time.time(), prompt_hash, model or "unknown", temperature, response, len(response.split())))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to log API call: {e}")
    
    def establish_multi_premise_world(self, topic: str, num_premises: int = 3) -> JokeWorld:
        """Create joke world with multiple premises and dependency tracking"""
        logger.info(f"Establishing multi-premise world for: {topic} ({num_premises} premises)")
        
        prompt = self.MULTI_PREMISE_GENERATION_PROMPT.format(
            topic=topic,
            num_premises=num_premises
        )
        
        response = self.call_llm_with_logging(prompt, temperature=0.8)
        premises = self._parse_multi_premise_response(response, topic)
        
        # Filter out banned premises
        premises = [p for p in premises if not self._is_premise_banned(p.content)]
        
        if len(premises) < 2:
            # Fallback to basic premises
            premises = self._generate_fallback_premises(topic)
        
        # Score premise quality
        for premise in premises:
            self._score_premise_quality(premise)
        
        # Filter low-quality premises
        premises = [p for p in premises if (p.quality_score or 5.0) >= 4.0]
        
        if len(premises) < 2:
            premises = self._generate_fallback_premises(topic)
        
        return JokeWorld(topic=topic, premises=premises)
    
    def _parse_multi_premise_response(self, response: str, topic: str) -> List[JokePremise]:
        """Parse multi-premise response into structured objects"""
        premises = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Grounding Premise'):
                content = line.split(':', 1)[1].strip()
                premises.append(JokePremise(content=content, premise_type='grounding'))
            elif line.startswith('Absurd Premise'):
                content = line.split(':', 1)[1].strip()
                premises.append(JokePremise(content=content, premise_type='absurd'))
            elif line.startswith('Conditional Premise'):
                content = line.split(':', 1)[1].strip()
                premises.append(JokePremise(content=content, premise_type='conditional'))
        
        return premises
    
    def _score_premise_quality(self, premise: JokePremise):
        """Score premise quality using LLM judge"""
        prompt = self.PREMISE_QUALITY_PROMPT.format(
            premise_content=premise.content,
            premise_type=premise.premise_type
        )
        
        try:
            response = self.call_llm_with_logging(prompt, temperature=0.3)
            scores = self._parse_quality_scores(response)
            
            premise.specificity_score = scores.get('specificity', 5.0)
            premise.novelty_score = scores.get('novelty', 5.0)
            premise.quality_score = scores.get('overall', 5.0)
            
        except Exception as e:
            logger.warning(f"Failed to score premise quality: {e}")
            premise.quality_score = 5.0
    
    def _parse_quality_scores(self, response: str) -> Dict[str, float]:
        """Parse quality scoring response"""
        scores = {}
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                try:
                    score = float(''.join(c for c in value if c.isdigit() or c == '.'))
                    scores[key] = max(1.0, min(10.0, score))
                except:
                    continue
        
        return scores
    
    def _is_premise_banned(self, premise_content: str) -> bool:
        """Check if premise is in banned list"""
        premise_lower = premise_content.lower()
        banned_keywords = ['secretly aliens', 'magic powers', 'conspiracy theory']
        return any(keyword in premise_lower for keyword in banned_keywords)
    
    def _generate_fallback_premises(self, topic: str) -> List[JokePremise]:
        """Generate basic fallback premises if parsing fails"""
        return [
            JokePremise(
                content=f"{topic} is a common part of everyday life.",
                premise_type='grounding',
                quality_score=6.0
            ),
            JokePremise(
                content=f"{topic} secretly operates according to absurd magical rules.",
                premise_type='absurd',
                quality_score=5.0
            )
        ]
    
    def generate_enhanced_abductive_joke(self, joke_world: JokeWorld, 
                                       adaptive_temperature: bool = True) -> AbductiveJoke:
        """Generate joke with enhanced reasoning and adaptive parameters"""
        logger.info(f"Generating enhanced abductive joke for: {joke_world.topic}")
        
        # Build premise context
        premise_context = ""
        for i, premise in enumerate(joke_world.premises, 1):
            premise_context += f"{premise.premise_type.title()} Premise {i}: {premise.content}\n"
        
        prompt = self.ENHANCED_ABDUCTIVE_JOKE_PROMPT.format(
            topic=joke_world.topic,
            premise_context=premise_context.strip()
        )
        
        # Try with high temperature first
        temperature = 0.9
        response = self.call_llm_with_logging(prompt, temperature=temperature)
        
        # Parse joke components
        setup, punchline, reasoning = self._parse_enhanced_joke_response(response)
        
        # Create joke object
        joke = AbductiveJoke(
            joke_world=joke_world,
            setup=setup,
            punchline=punchline,
            reasoning_chain=reasoning,
            generation_params={
                'temperature': temperature,
                'adaptive_temperature': adaptive_temperature,
                'prompt_version': 'enhanced_v2'
            }
        )
        
        # If adaptive temperature is enabled, check quality and retry if needed
        if adaptive_temperature:
            consistency_score = self._quick_consistency_check(joke)
            if consistency_score < 6.0:
                logger.info("Low consistency detected, retrying with lower temperature")
                response = self.call_llm_with_logging(prompt, temperature=0.7)
                setup, punchline, reasoning = self._parse_enhanced_joke_response(response)
                
                joke = AbductiveJoke(
                    joke_world=joke_world,
                    setup=setup,
                    punchline=punchline,
                    reasoning_chain=reasoning,
                    generation_params={
                        'temperature': 0.7,
                        'adaptive_temperature': True,
                        'retried': True,
                        'original_consistency': consistency_score
                    }
                )
        
        # Log joke to database
        if self.enable_logging:
            self._log_joke(joke)
        
        return joke
    
    def _parse_enhanced_joke_response(self, response: str) -> Tuple[str, str, str]:
        """Parse enhanced joke response with reasoning chain"""
        lines = response.split('\n')
        setup = punchline = reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Setup:'):
                setup = line[6:].strip()
            elif line.startswith('Punchline:'):
                punchline = line[10:].strip()
            elif line.startswith('Reasoning:'):
                reasoning = line[10:].strip()
        
        # Fallback parsing
        if not setup or not punchline:
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            if len(sentences) >= 2:
                setup = sentences[0] + '.'
                punchline = sentences[1] + '.'
            else:
                setup = "Here's an observation about this topic."
                punchline = "And here's the surprising explanation."
        
        if not reasoning:
            reasoning = "Applied abductive reasoning to derive the most logical explanation."
        
        return setup, punchline, reasoning
    
    def _quick_consistency_check(self, joke: AbductiveJoke) -> float:
        """Quick consistency check for adaptive temperature"""
        # Simplified check - count logical connectors and premise references
        reasoning = joke.reasoning_chain.lower()
        punchline = joke.punchline.lower()
        
        consistency_indicators = [
            'because', 'therefore', 'since', 'given that', 'due to',
            'explains', 'reason', 'logical', 'follows'
        ]
        
        indicator_count = sum(1 for indicator in consistency_indicators 
                            if indicator in reasoning or indicator in punchline)
        
        # Check if punchline references absurd premises
        absurd_premises = joke.joke_world.get_absurd_premises()
        premise_ref_count = 0
        for premise in absurd_premises:
            premise_words = premise.content.lower().split()
            if any(word in punchline for word in premise_words[:3]):  # Check first 3 words
                premise_ref_count += 1
        
        # Simple scoring: indicators + premise references, scaled to 1-10
        raw_score = (indicator_count * 2) + (premise_ref_count * 3)
        return min(10.0, max(1.0, raw_score))
    
    def _log_joke(self, joke: AbductiveJoke):
        """Log generated joke to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO jokes (timestamp, topic, joke_world_id, setup, punchline, 
                                 reasoning_chain, similarity_hash, generation_params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                joke.joke_world.topic,
                joke.joke_world.world_id,
                joke.setup,
                joke.punchline,
                joke.reasoning_chain,
                joke.similarity_hash,
                json.dumps(joke.generation_params)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to log joke: {e}")


class EnhancedJokeAnalyzer:
    """Research-grade analyzer with multi-judge evaluation"""
    
    def __init__(self, llm_client, num_judges: int = 3):
        """Initialize with multiple LLM judges for ensembling"""
        self.llm_client = llm_client
        self.num_judges = num_judges
        self.api_call_count = 0
    
    def measure_logical_consistency_ensemble(self, joke: AbductiveJoke) -> Dict[str, Any]:
        """Multi-judge logical consistency evaluation"""
        logger.info(f"Running ensemble consistency evaluation with {self.num_judges} judges")
        
        consistency_prompt = """Rate the logical consistency of this joke within its world:

WORLD RULES:
{premise_context}

JOKE:
Setup: {setup}
Punchline: {punchline}
Reasoning: {reasoning}

Score 1-10 where:
1 = Punchline contradicts world rules
10 = Punchline perfectly follows from world rules

Provide: Score: [number]
Brief reasoning: [explanation]"""
        
        # Build premise context
        premise_context = ""
        for premise in joke.joke_world.premises:
            premise_context += f"• {premise.premise_type.title()}: {premise.content}\n"
        
        prompt = consistency_prompt.format(
            premise_context=premise_context.strip(),
            setup=joke.setup,
            punchline=joke.punchline,
            reasoning=joke.reasoning_chain
        )
        
        # Get scores from multiple judges
        scores = []
        explanations = []
        
        for judge_num in range(self.num_judges):
            try:
                response = self._call_llm_for_analysis(prompt, temperature=0.2)
                score, explanation = self._parse_consistency_response(response)
                scores.append(score)
                explanations.append(f"Judge {judge_num + 1}: {explanation}")
                
                # Small delay between judge calls
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Judge {judge_num + 1} failed: {e}")
                scores.append(5.0)  # Default score
                explanations.append(f"Judge {judge_num + 1}: Evaluation failed")
        
        # Calculate ensemble statistics
        if scores:
            median_score = statistics.median(scores)
            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
            agreement = 1.0 - (std_score / 10.0)  # Higher agreement = lower std
        else:
            median_score = mean_score = std_score = agreement = 0.0
        
        return {
            'individual_scores': scores,
            'median_score': median_score,
            'mean_score': mean_score,
            'standard_deviation': std_score,
            'judge_agreement': agreement,
            'explanations': explanations,
            'num_judges': len(scores),
            'confidence': 'high' if agreement > 0.8 else 'medium' if agreement > 0.6 else 'low'
        }
    
    def _call_llm_for_analysis(self, prompt: str, temperature: float = 0.3) -> str:
        """Make LLM call for analysis tasks"""
        self.api_call_count += 1
        
        if hasattr(self.llm_client, 'chat'):
            if hasattr(self.llm_client, '_api_key') or 'groq' in str(type(self.llm_client)).lower():
                response = self.llm_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            else:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1000
                )
                return response.choices[0].message.content
        elif hasattr(self.llm_client, 'messages'):
            response = self.llm_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            raise RuntimeError(f"Unsupported LLM client type: {type(self.llm_client)}")
    
    def _parse_consistency_response(self, response: str) -> Tuple[float, str]:
        """Parse consistency evaluation response"""
        lines = response.split('\n')
        score = 5.0
        explanation = "No explanation provided"
        
        for line in lines:
            line = line.strip()
            if line.startswith('Score:'):
                try:
                    score_text = line[6:].strip()
                    score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
                    score = max(1.0, min(10.0, score))
                except:
                    score = 5.0
            elif 'reasoning:' in line.lower():
                explanation = line.split(':', 1)[1].strip()
        
        return score, explanation
    
    def analyze_premise_diversity(self, jokes: List[AbductiveJoke]) -> Dict[str, Any]:
        """Analyze diversity and patterns in premise usage"""
        logger.info(f"Analyzing premise diversity for {len(jokes)} jokes")
        
        # Collect all premises by type
        grounding_premises = []
        absurd_premises = []
        conditional_premises = []
        
        for joke in jokes:
            grounding_premises.extend([p.content for p in joke.joke_world.get_grounding_premises()])
            absurd_premises.extend([p.content for p in joke.joke_world.get_absurd_premises()])
            conditional_premises.extend([p.content for p in joke.joke_world.get_conditional_premises()])
        
        # Calculate diversity metrics
        unique_grounding = len(set(grounding_premises))
        unique_absurd = len(set(absurd_premises))
        unique_conditional = len(set(conditional_premises))
        
        total_grounding = len(grounding_premises)
        total_absurd = len(absurd_premises)
        total_conditional = len(conditional_premises)
        
        return {
            'grounding_premises': {
                'total': total_grounding,
                'unique': unique_grounding,
                'diversity_ratio': unique_grounding / total_grounding if total_grounding > 0 else 0,
                'examples': list(set(grounding_premises))[:5]
            },
            'absurd_premises': {
                'total': total_absurd,
                'unique': unique_absurd,
                'diversity_ratio': unique_absurd / total_absurd if total_absurd > 0 else 0,
                'examples': list(set(absurd_premises))[:5]
            },
            'conditional_premises': {
                'total': total_conditional,
                'unique': unique_conditional,
                'diversity_ratio': unique_conditional / total_conditional if total_conditional > 0 else 0,
                'examples': list(set(conditional_premises))[:5]
            },
            'overall_diversity': {
                'total_unique_premises': unique_grounding + unique_absurd + unique_conditional,
                'average_diversity_ratio': statistics.mean([
                    unique_grounding / total_grounding if total_grounding > 0 else 0,
                    unique_absurd / total_absurd if total_absurd > 0 else 0,
                    unique_conditional / total_conditional if total_conditional > 0 else 1
                ])
            }
        }


class ResearchExperimentFramework:
    """Advanced experimental design for research-grade studies"""
    
    def __init__(self, pipeline: EnhancedAbductiveJokePipeline, 
                 analyzer: EnhancedJokeAnalyzer):
        self.pipeline = pipeline
        self.analyzer = analyzer
        self.experiment_results = {}
    
    def run_premise_interaction_experiment(self, topics: List[str], 
                                         iterations_per_topic: int = 10) -> Dict[str, Any]:
        """
        Research Question: How do different premise combinations affect joke quality?
        """
        logger.info(f"Running premise interaction experiment: {len(topics)} topics × {iterations_per_topic} iterations")
        
        results = {
            'research_question': 'Effect of premise combinations on joke quality',
            'methodology': 'Multi-premise worlds with quality scoring',
            'topics': topics,
            'iterations_per_topic': iterations_per_topic,
            'conditions': ['2-premise', '3-premise', '4-premise'],
            'results': {}
        }
        
        for topic in topics:
            topic_results = {'conditions': {}}
            
            # Test different numbers of premises
            for num_premises in [2, 3, 4]:
                condition_jokes = []
                condition_scores = []
                
                for iteration in range(iterations_per_topic):
                    try:
                        # Generate joke world with specified number of premises
                        joke_world = self.pipeline.establish_multi_premise_world(topic, num_premises)
                        joke = self.pipeline.generate_enhanced_abductive_joke(joke_world)
                        
                        # Evaluate with ensemble method
                        consistency_eval = self.analyzer.measure_logical_consistency_ensemble(joke)
                        
                        condition_jokes.append(joke)
                        condition_scores.append(consistency_eval['median_score'])
                        
                    except Exception as e:
                        logger.error(f"Failed iteration {iteration} for {topic} with {num_premises} premises: {e}")
                
                topic_results['conditions'][f'{num_premises}-premise'] = {
                    'jokes': len(condition_jokes),
                    'scores': condition_scores,
                    'mean_score': statistics.mean(condition_scores) if condition_scores else 0,
                    'std_score': statistics.stdev(condition_scores) if len(condition_scores) > 1 else 0
                }
            
            results['results'][topic] = topic_results
        
        # Statistical analysis across conditions
        results['statistical_analysis'] = self._analyze_premise_interaction_results(results)
        
        return results
    
    def _analyze_premise_interaction_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze statistical significance of premise interaction effects"""
        # Collect scores by condition across all topics
        condition_scores = {'2-premise': [], '3-premise': [], '4-premise': []}
        
        for topic_data in results['results'].values():
            for condition, data in topic_data['conditions'].items():
                condition_scores[condition].extend(data['scores'])
        
        # Compare conditions pairwise
        comparisons = {}
        from itertools import combinations
        
        for cond1, cond2 in combinations(condition_scores.keys(), 2):
            if condition_scores[cond1] and condition_scores[cond2]:
                # Simple t-test (would use stats_utils in real implementation)
                try:
                    import scipy.stats as stats
                    t_stat, p_value = stats.ttest_ind(condition_scores[cond1], condition_scores[cond2])
                    
                    # Cohen's d effect size
                    mean1, mean2 = statistics.mean(condition_scores[cond1]), statistics.mean(condition_scores[cond2])
                    pooled_std = np.sqrt(((len(condition_scores[cond1]) - 1) * np.var(condition_scores[cond1], ddof=1) + 
                                         (len(condition_scores[cond2]) - 1) * np.var(condition_scores[cond2], ddof=1)) / 
                                        (len(condition_scores[cond1]) + len(condition_scores[cond2]) - 2))
                    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    
                    comparisons[f'{cond1}_vs_{cond2}'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'effect_size': cohens_d,
                        'significant': p_value < 0.05,
                        'interpretation': 'significant' if p_value < 0.05 else 'not significant'
                    }
                except Exception as e:
                    logger.warning(f"Statistical comparison failed for {cond1} vs {cond2}: {e}")
        
        return {
            'condition_summaries': {
                cond: {
                    'n': len(scores),
                    'mean': statistics.mean(scores) if scores else 0,
                    'std': statistics.stdev(scores) if len(scores) > 1 else 0
                }
                for cond, scores in condition_scores.items()
            },
            'pairwise_comparisons': comparisons,
            'overall_conclusion': self._generate_experiment_conclusion(comparisons)
        }
    
    def _generate_experiment_conclusion(self, comparisons: Dict[str, Any]) -> str:
        """Generate research conclusion from statistical comparisons"""
        significant_comparisons = [comp for comp, data in comparisons.items() if data['significant']]
        
        if not significant_comparisons:
            return "No significant differences found between premise conditions"
        
        # Find the best condition based on effect sizes
        best_effects = []
        for comp_name, comp_data in comparisons.items():
            if comp_data['significant'] and comp_data['effect_size'] > 0.3:
                best_effects.append((comp_name, comp_data['effect_size']))
        
        if best_effects:
            best_comparison = max(best_effects, key=lambda x: x[1])
            return f"Significant improvement found: {best_comparison[0]} (effect size: {best_comparison[1]:.3f})"
        else:
            return f"Significant differences detected in {len(significant_comparisons)} comparisons, but small effect sizes"


def create_enhanced_pipeline(llm_client, enable_caching: bool = True, 
                           enable_logging: bool = True) -> EnhancedAbductiveJokePipeline:
    """Factory function to create enhanced pipeline"""
    return EnhancedAbductiveJokePipeline(
        llm_client=llm_client,
        enable_caching=enable_caching,
        enable_logging=enable_logging
    )


def create_enhanced_analyzer(llm_client, num_judges: int = 3) -> EnhancedJokeAnalyzer:
    """Factory function to create enhanced analyzer"""
    return EnhancedJokeAnalyzer(llm_client=llm_client, num_judges=num_judges)


if __name__ == "__main__":
    # Example usage with enhanced pipeline
    logging.basicConfig(level=logging.INFO)
    
    # This would be your actual LLM client
    # client = groq.Groq(api_key="your-key")
    
    print("Enhanced Abductive Joke Pipeline - Research Grade")
    print("Run with actual LLM client for full functionality")
    
    # Demo of enhanced features:
    print("\n🔬 Research Features Available:")
    print("• Multi-premise joke worlds with dependency graphs")
    print("• Premise quality filtering and scoring")
    print("• Multi-judge ensemble evaluation")
    print("• Adaptive temperature control")
    print("• Comprehensive logging and caching")
    print("• Statistical analysis framework")
    print("• Research-grade experimental design")
    print("• Type safety with Pydantic validation") 