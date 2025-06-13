"""
JokePlanSearch: A Comprehensive LLM-Based Joke Generation and Evaluation System

This module implements the PlanSearch methodology for computational humor generation,
incorporating research-backed bias mitigation techniques for LLM-as-a-judge evaluation.

Key Features:
- Multi-stage joke generation with diverse angle exploration
- Bias-minimized LLM evaluation with ensemble judging
- Comprehensive scoring across multiple humor dimensions
- Rate limiting and robust error handling for API calls
"""

import json
import time
import random
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import os
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
                    # Generic requests-based approach
                    raise NotImplementedError("Please implement API client integration")
                
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
    
    def analyze_topic(self, topic: str) -> TopicAnalysis:
        """
        Phase 2.1: Analyze the input topic for joke generation opportunities.
        
        Args:
            topic: The input topic to analyze
            
        Returns:
            TopicAnalysis object with contextual information
        """
        logger.info(f"Analyzing topic: {topic}")
        
        analysis_prompt = f"""Analyze this topic for joke-writing potential: "{topic}"

Provide your analysis in the following format:
Context: [What is this topic about? What domain/field does it relate to?]
Wordplay Opportunities: [What words, sounds, or phrases could be used for puns or wordplay?]
Cultural References: [What pop culture, common knowledge, or stereotypes are associated with this topic?]
Absurdity Potential: [What unexpected or surreal angles could work with this topic?]"""

        response = self.call_llm(analysis_prompt)
        
        # Parse the structured response
        lines = response.split('\n')
        analysis_data = {
            'context_analysis': '',
            'wordplay_potential': '',
            'cultural_references': '',
            'absurdity_potential': ''
        }
        
        current_field = None
        for line in lines:
            line = line.strip()
            if line.startswith('Context:'):
                current_field = 'context_analysis'
                analysis_data[current_field] = line[8:].strip()
            elif line.startswith('Wordplay Opportunities:'):
                current_field = 'wordplay_potential'
                analysis_data[current_field] = line[23:].strip()
            elif line.startswith('Cultural References:'):
                current_field = 'cultural_references'
                analysis_data[current_field] = line[19:].strip()
            elif line.startswith('Absurdity Potential:'):
                current_field = 'absurdity_potential'
                analysis_data[current_field] = line[19:].strip()
            elif current_field and line:
                analysis_data[current_field] += ' ' + line
        
        self.topic_analysis = TopicAnalysis(
            original_topic=topic,
            **analysis_data
        )
        
        logger.info("Topic analysis completed")
        return self.topic_analysis
    
    def generate_diverse_joke_angles(self) -> List[str]:
        """
        Phase 2.2: Generate diverse joke angles using multiple humor categories.
        
        Returns:
            List of diverse joke angles
        """
        if not self.topic_analysis:
            raise ValueError("Must analyze topic before generating angles")
            
        logger.info("Generating diverse joke angles")
        topic = self.topic_analysis.original_topic
        
        # Basic angles prompt
        basic_prompt = f"""Generate 5 distinct joke angles for the topic: "{topic}"

Consider these humor categories:
- Puns and wordplay
- Observational humor (what's funny about everyday situations)
- Absurdist/surreal humor
- Character-based humor (stereotypes, personas)
- Situational irony

Format your response as:
1. [Angle 1]: [Brief description of the approach]
2. [Angle 2]: [Brief description of the approach]
3. [Angle 3]: [Brief description of the approach]
4. [Angle 4]: [Brief description of the approach]
5. [Angle 5]: [Brief description of the approach]"""

        # Advanced techniques prompt
        advanced_prompt = f"""For the topic "{topic}", generate 3 sophisticated joke concepts using these advanced techniques:

- Misdirection (set up one expectation, deliver something unexpected)
- Meta-humor (jokes about jokes or breaking the fourth wall)
- Callback structure (reference something earlier in the setup)

Format as:
1. [Technique]: [Specific angle for "{topic}"]
2. [Technique]: [Specific angle for "{topic}"]
3. [Technique]: [Specific angle for "{topic}"]"""

        # Combination strategy prompt
        combination_prompt = f"""Create 3 hybrid joke concepts by combining "{topic}" with unexpected elements:

Topic Analysis: {self.topic_analysis.context_analysis}

Generate combinations with:
1. An unrelated everyday object or situation
2. A completely different professional field or expertise area  
3. A different time period or cultural context

Format as:
1. Hybrid: [Description of the combination approach]
2. Hybrid: [Description of the combination approach]
3. Hybrid: [Description of the combination approach]"""

        # Make all three calls in parallel for efficiency
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.call_llm, basic_prompt): 'basic',
                executor.submit(self.call_llm, advanced_prompt): 'advanced',
                executor.submit(self.call_llm, combination_prompt): 'combination'
            }
            
            responses = {}
            for future in as_completed(futures):
                response_type = futures[future]
                try:
                    responses[response_type] = future.result()
                except Exception as e:
                    logger.error(f"Failed to get {response_type} angles: {str(e)}")
                    responses[response_type] = ""
        
        # Parse all responses to extract angles
        all_angles = []
        
        for response_type, response in responses.items():
            angles = self._parse_numbered_list(response)
            all_angles.extend(angles)
            logger.info(f"Extracted {len(angles)} {response_type} angles")
        
        # Deduplicate similar angles
        unique_angles = self._deduplicate_angles(all_angles)
        
        self.joke_angles = unique_angles
        logger.info(f"Generated {len(unique_angles)} unique joke angles")
        
        return unique_angles
    
    def _parse_numbered_list(self, response: str) -> List[str]:
        """Parse a numbered list response into individual items."""
        angles = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered items (1., 2., etc.)
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and extract the angle description
                if ':' in line:
                    angle = line.split(':', 1)[1].strip()
                else:
                    # Remove leading number/bullet and take the rest
                    angle = line.lstrip('0123456789.-) ').strip()
                
                if angle:
                    angles.append(angle)
        
        return angles
    
    def _deduplicate_angles(self, angles: List[str]) -> List[str]:
        """Remove similar angles using simple keyword overlap."""
        unique_angles = []
        
        for angle in angles:
            is_duplicate = False
            angle_words = set(angle.lower().split())
            
            for existing_angle in unique_angles:
                existing_words = set(existing_angle.lower().split())
                # If more than 60% of words overlap, consider it a duplicate
                overlap = len(angle_words & existing_words) / len(angle_words | existing_words)
                if overlap > 0.6:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_angles.append(angle)
        
        return unique_angles
    
    def generate_jokes_from_angles(self) -> List[JokeCandidate]:
        """
        Phase 2.3: Generate full jokes from each angle using structured execution.
        
        Returns:
            List of JokeCandidate objects with initial jokes
        """
        if not self.joke_angles:
            raise ValueError("Must generate joke angles before creating jokes")
            
        logger.info(f"Generating jokes from {len(self.joke_angles)} angles")
        
        joke_candidates = []
        topic = self.topic_analysis.original_topic
        context = self.topic_analysis.context_analysis
        
        # Generate jokes in parallel for efficiency
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_angle = {}
            
            for angle in self.joke_angles:
                prompt = f"""Create a joke using this approach:
Topic: "{topic}"
Angle: "{angle}"
Context: {context}

Step 1: Write a brief outline of your joke structure (setup, punchline, any callbacks)
Step 2: Write the complete joke

Outline: [Your joke structure here]
Joke: [Your complete joke here]"""
                
                future = executor.submit(self.call_llm, prompt)
                future_to_angle[future] = angle
            
            for future in as_completed(future_to_angle):
                angle = future_to_angle[future]
                try:
                    response = future.result()
                    outline, joke = self._parse_joke_response(response)
                    
                    candidate = JokeCandidate(
                        angle=angle,
                        outline=outline,
                        full_joke=joke
                    )
                    joke_candidates.append(candidate)
                    logger.info(f"Generated joke for angle: {angle[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Failed to generate joke for angle '{angle}': {str(e)}")
        
        self.joke_candidates = joke_candidates
        logger.info(f"Successfully generated {len(joke_candidates)} jokes")
        
        return joke_candidates
    
    def _parse_joke_response(self, response: str) -> Tuple[str, str]:
        """Parse the structured joke generation response."""
        outline = ""
        joke = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Outline:'):
                current_section = 'outline'
                outline = line[8:].strip()
            elif line.startswith('Joke:'):
                current_section = 'joke'
                joke = line[5:].strip()
            elif current_section == 'outline' and line:
                outline += ' ' + line
            elif current_section == 'joke' and line:
                joke += ' ' + line
        
        return outline.strip(), joke.strip()
    
    def refine_jokes(self, refinement_rounds: int = 2) -> None:
        """
        Phase 2.4: Quality improvement through multi-stage refinement.
        
        Args:
            refinement_rounds: Number of refinement iterations to perform
        """
        if not self.joke_candidates:
            raise ValueError("Must generate jokes before refining them")
            
        logger.info(f"Refining {len(self.joke_candidates)} jokes through {refinement_rounds} rounds")
        
        for round_num in range(refinement_rounds):
            logger.info(f"Starting refinement round {round_num + 1}")
            
            # For first round, refine all jokes. For subsequent rounds, only top 50%
            if round_num == 0:
                candidates_to_refine = self.joke_candidates
            else:
                # Sort by average score and take top 50%
                scored_candidates = [c for c in self.joke_candidates if c.scores]
                if scored_candidates:
                    scored_candidates.sort(key=lambda x: statistics.mean(x.scores.values()), reverse=True)
                    candidates_to_refine = scored_candidates[:len(scored_candidates)//2]
                else:
                    candidates_to_refine = self.joke_candidates[:len(self.joke_candidates)//2]
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                
                for candidate in candidates_to_refine:
                    future = executor.submit(self._refine_single_joke, candidate)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        future.result()  # This updates the candidate in place
                    except Exception as e:
                        logger.error(f"Failed to refine joke: {str(e)}")
            
            logger.info(f"Completed refinement round {round_num + 1}")
    
    def _refine_single_joke(self, candidate: JokeCandidate) -> None:
        """Refine a single joke through critique and improvement."""
        topic = self.topic_analysis.original_topic
        current_joke = candidate.refined_joke or candidate.full_joke
        
        # Critique prompt
        critique_prompt = f"""Analyze this joke for improvement opportunities:
"{current_joke}"

Provide specific feedback on:
1. Setup clarity: Is the setup clear and economical?
2. Punchline impact: Is the punchline surprising and satisfying?
3. Timing: Are there unnecessary words that hurt the rhythm?
4. Relevance: Does it clearly connect to the original topic "{topic}"?

Critique: [Your detailed analysis]
Improvement Suggestions: [Specific changes to make it funnier]"""

        critique_response = self.call_llm(critique_prompt)
        critique, suggestions = self._parse_critique_response(critique_response)
        
        # Refinement prompt
        refinement_prompt = f"""Original joke: "{current_joke}"
Critique: "{critique}"
Improvement suggestions: "{suggestions}"

Now write an improved version of the joke that addresses these issues. Make it funnier, tighter, and more impactful.

Improved Joke: [Your refined joke here]"""

        refinement_response = self.call_llm(refinement_prompt)
        refined_joke = self._extract_refined_joke(refinement_response)
        
        # Update candidate
        candidate.critique = critique
        candidate.improvement_suggestions = suggestions
        candidate.refined_joke = refined_joke
    
    def _parse_critique_response(self, response: str) -> Tuple[str, str]:
        """Parse critique response into critique and suggestions."""
        critique = ""
        suggestions = ""
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Critique:'):
                current_section = 'critique'
                critique = line[9:].strip()
            elif line.startswith('Improvement Suggestions:'):
                current_section = 'suggestions'
                suggestions = line[23:].strip()
            elif current_section == 'critique' and line:
                critique += ' ' + line
            elif current_section == 'suggestions' and line:
                suggestions += ' ' + line
        
        return critique.strip(), suggestions.strip()
    
    def _extract_refined_joke(self, response: str) -> str:
        """Extract the refined joke from the response."""
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Improved Joke:'):
                return line[14:].strip()
        
        # If no specific marker found, return the whole response
        return response.strip()
    
    def ensure_joke_diversity(self, similarity_threshold: float = 0.7) -> None:
        """
        Phase 2.5: Enhance joke diversity by detecting and replacing similar jokes.
        
        Args:
            similarity_threshold: Threshold for considering jokes too similar
        """
        if not self.joke_candidates:
            return
            
        logger.info("Analyzing joke diversity and generating additional jokes if needed")
        
        # Group similar jokes
        similar_groups = self._find_similar_joke_groups(similarity_threshold)
        
        if len(similar_groups) > 1:
            logger.info(f"Found {len(similar_groups)} groups of similar jokes")
            
            # Get examples of existing jokes for diversity prompt
            existing_jokes = [candidate.refined_joke or candidate.full_joke 
                            for candidate in self.joke_candidates[:5]]
            
            diversity_prompt = f"""I have generated several jokes about "{self.topic_analysis.original_topic}" but they seem too similar. Create 3 jokes that are completely different in style and approach from these existing ones:

Existing jokes:
{chr(10).join(f"- {joke}" for joke in existing_jokes)}

Generate jokes that use different:
- Humor mechanisms (if existing ones use puns, try observational humor)
- Perspectives (if existing ones are first-person, try third-person or objective)
- Formats (if existing ones are one-liners, try short stories or dialogues)

New Joke 1: [Completely different approach]
New Joke 2: [Completely different approach]  
New Joke 3: [Completely different approach]"""

            try:
                response = self.call_llm(diversity_prompt)
                new_jokes = self._parse_numbered_list(response)
                
                # Add new diverse jokes as candidates
                for i, joke in enumerate(new_jokes):
                    if joke:
                        candidate = JokeCandidate(
                            angle=f"Diversity enhancement #{i+1}",
                            outline="Generated for diversity",
                            full_joke=joke,
                            refined_joke=joke
                        )
                        self.joke_candidates.append(candidate)
                
                logger.info(f"Added {len(new_jokes)} diverse jokes")
                
            except Exception as e:
                logger.error(f"Failed to generate diverse jokes: {str(e)}")
    
    def _find_similar_joke_groups(self, threshold: float) -> List[List[int]]:
        """Group jokes by similarity using simple keyword overlap."""
        groups = []
        used_indices = set()
        
        for i, candidate_i in enumerate(self.joke_candidates):
            if i in used_indices:
                continue
                
            joke_i = candidate_i.refined_joke or candidate_i.full_joke
            words_i = set(joke_i.lower().split())
            current_group = [i]
            used_indices.add(i)
            
            for j, candidate_j in enumerate(self.joke_candidates[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                joke_j = candidate_j.refined_joke or candidate_j.full_joke
                words_j = set(joke_j.lower().split())
                
                # Calculate similarity
                overlap = len(words_i & words_j) / len(words_i | words_j) if words_i | words_j else 0
                
                if overlap > threshold:
                    current_group.append(j)
                    used_indices.add(j)
            
            groups.append(current_group)
        
        return groups

    def evaluate_jokes_multidimensional(self) -> Dict[str, Any]:
        """
        Phase 3.2: Multi-dimensional evaluation of jokes across humor criteria.
        
        Returns:
            Dictionary containing evaluation results and metrics
        """
        if not self.joke_candidates:
            raise ValueError("Must generate jokes before evaluating them")
            
        logger.info("Starting multi-dimensional joke evaluation")
        
        # Shuffle jokes for positional bias mitigation
        shuffled_candidates = self.joke_candidates.copy()
        random.shuffle(shuffled_candidates)
        
        # Prepare jokes list for evaluation
        jokes_list = []
        for i, candidate in enumerate(shuffled_candidates):
            joke_text = candidate.refined_joke or candidate.full_joke
            jokes_list.append(f"{i+1}. {joke_text}")
        
        evaluation_prompt = f"""You are an expert comedy analyst. Rate each joke on these 5 dimensions using a scale of 1-10:

1. CLEVERNESS: How intellectually satisfying is the humor?
2. SURPRISE: How unexpected is the punchline?
3. RELATABILITY: How well will audiences connect with this?
4. TIMING/RHYTHM: How well does the joke flow and land?
5. OVERALL FUNNINESS: Your general assessment of how funny this is

Here are the jokes to evaluate (in random order):
{chr(10).join(jokes_list)}

For each joke, provide ratings in this exact format:
JOKE: [joke text]
CLEVERNESS: [score 1-10]
SURPRISE: [score 1-10]
RELATABILITY: [score 1-10]
TIMING: [score 1-10]
OVERALL: [score 1-10]
REASONING: [brief explanation of scores]
---"""

        try:
            response = self.call_llm_judge(evaluation_prompt)
            scores = self._parse_multidimensional_scores(response)
            
            # Map scores back to original candidates
            for i, candidate in enumerate(shuffled_candidates):
                if i < len(scores):
                    candidate.scores.update(scores[i])
            
            logger.info("Multi-dimensional evaluation completed")
            return {"evaluation_type": "multidimensional", "scores_count": len(scores)}
            
        except Exception as e:
            logger.error(f"Multi-dimensional evaluation failed: {str(e)}")
            return {"error": str(e)}
    
    def _parse_multidimensional_scores(self, response: str) -> List[Dict[str, float]]:
        """Parse multi-dimensional evaluation scores from LLM response."""
        scores = []
        current_joke_scores = {}
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('JOKE:'):
                # Start of new joke evaluation
                if current_joke_scores:
                    scores.append(current_joke_scores)
                current_joke_scores = {}
                
            elif line.startswith('CLEVERNESS:'):
                try:
                    score = float(line.split(':')[1].strip())
                    current_joke_scores['cleverness'] = score
                except (ValueError, IndexError):
                    current_joke_scores['cleverness'] = 5.0
                    
            elif line.startswith('SURPRISE:'):
                try:
                    score = float(line.split(':')[1].strip())
                    current_joke_scores['surprise'] = score
                except (ValueError, IndexError):
                    current_joke_scores['surprise'] = 5.0
                    
            elif line.startswith('RELATABILITY:'):
                try:
                    score = float(line.split(':')[1].strip())
                    current_joke_scores['relatability'] = score
                except (ValueError, IndexError):
                    current_joke_scores['relatability'] = 5.0
                    
            elif line.startswith('TIMING:'):
                try:
                    score = float(line.split(':')[1].strip())
                    current_joke_scores['timing'] = score
                except (ValueError, IndexError):
                    current_joke_scores['timing'] = 5.0
                    
            elif line.startswith('OVERALL:'):
                try:
                    score = float(line.split(':')[1].strip())
                    current_joke_scores['overall'] = score
                except (ValueError, IndexError):
                    current_joke_scores['overall'] = 5.0
            
            elif line == '---':
                # End of current joke evaluation
                if current_joke_scores:
                    scores.append(current_joke_scores)
                    current_joke_scores = {}
        
        # Don't forget the last joke if there's no trailing separator
        if current_joke_scores:
            scores.append(current_joke_scores)
        
        return scores
    
    def evaluate_jokes_comparative(self, min_comparisons: int = None) -> Dict[str, Any]:
        """
        Phase 3.3: Comparative evaluation using pairwise comparisons.
        
        Args:
            min_comparisons: Minimum comparisons per joke (defaults to config)
            
        Returns:
            Dictionary containing comparative evaluation results
        """
        if not self.joke_candidates:
            raise ValueError("Must generate jokes before evaluating them")
            
        if min_comparisons is None:
            min_comparisons = self.bias_config.min_comparisons_per_joke
            
        logger.info("Starting comparative joke evaluation")
        
        n_jokes = len(self.joke_candidates)
        total_comparisons = max(n_jokes * min_comparisons, n_jokes * 2)
        
        comparison_results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            for _ in range(total_comparisons):
                # Randomly select two different jokes
                idx_a, idx_b = random.sample(range(n_jokes), 2)
                candidate_a = self.joke_candidates[idx_a]
                candidate_b = self.joke_candidates[idx_b]
                
                joke_a = candidate_a.refined_joke or candidate_a.full_joke
                joke_b = candidate_b.refined_joke or candidate_b.full_joke
                
                future = executor.submit(self._compare_jokes, joke_a, joke_b, idx_a, idx_b)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    comparison_results.append(result)
                except Exception as e:
                    logger.error(f"Comparison failed: {str(e)}")
        
        # Process comparison results
        self._process_comparison_results(comparison_results)
        
        logger.info(f"Completed {len(comparison_results)} pairwise comparisons")
        return {
            "evaluation_type": "comparative", 
            "total_comparisons": len(comparison_results)
        }
    
    def _compare_jokes(self, joke_a: str, joke_b: str, idx_a: int, idx_b: int) -> Dict[str, Any]:
        """Compare two jokes and return the result."""
        comparison_prompt = f"""Compare these two jokes and determine which is funnier. Consider overall impact, cleverness, and how likely someone is to laugh.

JOKE A: "{joke_a}"
JOKE B: "{joke_b}"

Which joke is funnier and why?

WINNER: [A or B]
REASONING: [Explain why the winning joke is superior]
MARGIN: [How much funnier is it? "Slightly", "Moderately", or "Much funnier"]"""

        response = self.call_llm_judge(comparison_prompt)
        
        # Parse response
        winner = None
        reasoning = ""
        margin = ""
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('WINNER:'):
                winner_text = line[7:].strip().upper()
                winner = 'A' if 'A' in winner_text else 'B' if 'B' in winner_text else None
            elif line.startswith('REASONING:'):
                reasoning = line[10:].strip()
            elif line.startswith('MARGIN:'):
                margin = line[7:].strip()
        
        return {
            'idx_a': idx_a,
            'idx_b': idx_b,
            'winner': winner,
            'reasoning': reasoning,
            'margin': margin
        }
    
    def _process_comparison_results(self, comparison_results: List[Dict[str, Any]]) -> None:
        """Process pairwise comparison results to update win/loss records."""
        for result in comparison_results:
            idx_a = result['idx_a']
            idx_b = result['idx_b']
            winner = result['winner']
            
            if winner == 'A':
                self.joke_candidates[idx_a].comparative_wins += 1
                self.joke_candidates[idx_b].comparative_losses += 1
            elif winner == 'B':
                self.joke_candidates[idx_b].comparative_wins += 1
                self.joke_candidates[idx_a].comparative_losses += 1
    
    def evaluate_jokes_ensemble(self) -> Dict[str, Any]:
        """
        Phase 3.4: Ensemble evaluation with different judge perspectives.
        
        Returns:
            Dictionary containing ensemble evaluation results
        """
        if not self.joke_candidates:
            raise ValueError("Must generate jokes before evaluating them")
            
        logger.info("Starting ensemble evaluation with multiple judge perspectives")
        
        # Prepare jokes for evaluation
        jokes_list = []
        for i, candidate in enumerate(self.joke_candidates):
            joke_text = candidate.refined_joke or candidate.full_joke
            jokes_list.append(f"{i+1}. {joke_text}")
        
        # Casual audience perspective
        casual_prompt = f"""Imagine you're at a comedy club with friends. Rate how much each joke would make you and your friends laugh on a scale of 1-10:

{chr(10).join(jokes_list)}

Rate each joke considering:
- Would this get actual laughs in a real social setting?
- Is it the kind of joke people would remember and retell?
- Does it work for a general audience?

For each joke, provide:
JOKE [number]: [score 1-10]
REASONING: [brief explanation]"""

        # Comedy expert perspective
        expert_prompt = f"""As a professional comedy writer, evaluate these jokes for their technical craft and commercial potential:

{chr(10).join(jokes_list)}

Consider:
- Professional joke construction and timing
- Originality and freshness of the concept
- Potential for use in professional comedy contexts

For each joke, provide:
JOKE [number]: [score 1-10]
REASONING: [brief explanation]"""

        # Execute both evaluations in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            casual_future = executor.submit(self.call_llm_judge, casual_prompt)
            expert_future = executor.submit(self.call_llm_judge, expert_prompt)
            
            try:
                casual_response = casual_future.result()
                expert_response = expert_future.result()
                
                casual_scores = self._parse_simple_scores(casual_response)
                expert_scores = self._parse_simple_scores(expert_response)
                
                # Update candidates with ensemble scores
                for i, candidate in enumerate(self.joke_candidates):
                    if i < len(casual_scores):
                        candidate.scores['casual_audience'] = casual_scores[i]
                    if i < len(expert_scores):
                        candidate.scores['expert_judge'] = expert_scores[i]
                
                logger.info("Ensemble evaluation completed")
                return {
                    "evaluation_type": "ensemble",
                    "casual_scores": len(casual_scores),
                    "expert_scores": len(expert_scores)
                }
                
            except Exception as e:
                logger.error(f"Ensemble evaluation failed: {str(e)}")
                return {"error": str(e)}
    
    def _parse_simple_scores(self, response: str) -> List[float]:
        """Parse simple score format responses."""
        scores = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('JOKE') and ':' in line:
                try:
                    score_part = line.split(':')[1].strip()
                    # Extract just the number (handle various formats)
                    score_str = ''.join(c for c in score_part.split()[0] if c.isdigit() or c == '.')
                    if score_str:
                        score = float(score_str)
                        scores.append(max(1.0, min(10.0, score)))  # Clamp to 1-10 range
                except (ValueError, IndexError):
                    continue
        
        return scores
    
    def calculate_final_rankings(self) -> List[Tuple[JokeCandidate, float]]:
        """
        Phase 4.1: Calculate final rankings with confidence intervals.
        
        Returns:
            List of (candidate, final_score) tuples sorted by score
        """
        logger.info("Calculating final rankings and confidence intervals")
        
        ranked_candidates = []
        
        for candidate in self.joke_candidates:
            # Calculate weighted final score
            scores = []
            weights = []
            
            # Multidimensional scores (weight: 0.4)
            if 'overall' in candidate.scores:
                scores.append(candidate.scores['overall'])
                weights.append(0.4)
            
            # Ensemble scores (weight: 0.3 each)
            if 'casual_audience' in candidate.scores:
                scores.append(candidate.scores['casual_audience'])
                weights.append(0.3)
            
            if 'expert_judge' in candidate.scores:
                scores.append(candidate.scores['expert_judge'])
                weights.append(0.3)
            
            # Comparative performance (weight: 0.2)
            total_comparisons = candidate.comparative_wins + candidate.comparative_losses
            if total_comparisons > 0:
                win_rate = candidate.comparative_wins / total_comparisons
                comparative_score = win_rate * 10  # Scale to 1-10
                scores.append(comparative_score)
                weights.append(0.2)
            
            # Calculate weighted average
            if scores and weights:
                # Normalize weights to sum to 1
                weight_sum = sum(weights)
                normalized_weights = [w / weight_sum for w in weights]
                
                final_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
                
                # Calculate confidence interval based on score variance
                if len(scores) > 1:
                    score_variance = statistics.variance(scores)
                    confidence_range = 1.96 * (score_variance ** 0.5) / (len(scores) ** 0.5)  # 95% CI
                    candidate.confidence_interval = (
                        max(1.0, final_score - confidence_range),
                        min(10.0, final_score + confidence_range)
                    )
                else:
                    candidate.confidence_interval = (final_score - 0.5, final_score + 0.5)
                
                ranked_candidates.append((candidate, final_score))
            else:
                # No scores available, assign default
                candidate.confidence_interval = (4.5, 5.5)
                ranked_candidates.append((candidate, 5.0))
        
        # Sort by final score (descending)
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Final rankings calculated for {len(ranked_candidates)} jokes")
        return ranked_candidates
    
    def generate_final_analysis(self, top_n: int = 5) -> str:
        """
        Phase 4.2: Generate comprehensive analysis of results.
        
        Args:
            top_n: Number of top jokes to analyze in detail
            
        Returns:
            Formatted analysis string
        """
        ranked_jokes = self.calculate_final_rankings()
        
        if not ranked_jokes:
            return "No jokes available for analysis."
        
        # Prepare data for analysis
        top_jokes = ranked_jokes[:top_n]
        total_count = len(self.joke_candidates)
        top_score = top_jokes[0][1] if top_jokes else 0
        average_score = statistics.mean([score for _, score in ranked_jokes])
        
        # Create analysis prompt
        analysis_prompt = f"""Analyze the results of this joke generation experiment:

Topic: "{self.topic_analysis.original_topic}"
Total jokes generated: {total_count}
Top joke score: {top_score:.2f}
Average score: {average_score:.2f}

Top {len(top_jokes)} jokes:
{chr(10).join([f"{i+1}. {candidate.refined_joke or candidate.full_joke} (Score: {score:.2f})" for i, (candidate, score) in enumerate(top_jokes)])}

Provide insights on:
1. What made the top jokes successful?
2. What patterns do you notice in the most successful humor approaches?
3. How could this generation process be improved?

Analysis: [Your insights here]"""

        try:
            analysis = self.call_llm(analysis_prompt)
            return analysis
        except Exception as e:
            logger.error(f"Failed to generate analysis: {str(e)}")
            return f"Analysis generation failed: {str(e)}"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance and usage metrics for the session."""
        return {
            "api_calls_made": self.api_call_count,
            "total_tokens_used": self.total_tokens_used,
            "jokes_generated": len(self.joke_candidates),
            "angles_explored": len(self.joke_angles),
            "successful_evaluations": len([c for c in self.joke_candidates if c.scores])
        }
    
    def run_complete_pipeline(self, topic: str, **kwargs) -> Dict[str, Any]:
        """
        Run the complete JokePlanSearch pipeline from topic to final analysis.
        
        Args:
            topic: The input topic for joke generation
            **kwargs: Additional configuration options
            
        Returns:
            Dictionary containing complete results
        """
        try:
            logger.info(f"Starting complete JokePlanSearch pipeline for topic: {topic}")
            
            # Phase 1: Setup (already done in __init__)
            
            # Phase 2: Topic Analysis and Joke Generation
            self.analyze_topic(topic)
            self.generate_diverse_joke_angles()
            self.generate_jokes_from_angles()
            self.refine_jokes(kwargs.get('refinement_rounds', 2))
            self.ensure_joke_diversity()
            
            # Phase 3: Evaluation
            self.evaluate_jokes_multidimensional()
            self.evaluate_jokes_comparative()
            self.evaluate_jokes_ensemble()
            
            # Phase 4: Final Analysis
            ranked_jokes = self.calculate_final_rankings()
            analysis = self.generate_final_analysis(kwargs.get('top_n', 5))
            metrics = self.get_performance_metrics()
            
            # Compile results
            results = {
                "topic": topic,
                "topic_analysis": asdict(self.topic_analysis),
                "total_jokes_generated": len(self.joke_candidates),
                "ranked_jokes": [
                    {
                        "joke": candidate.refined_joke or candidate.full_joke,
                        "angle": candidate.angle,
                        "score": score,
                        "confidence_interval": candidate.confidence_interval,
                        "detailed_scores": candidate.scores
                    }
                    for candidate, score in ranked_jokes
                ],
                "analysis": analysis,
                "performance_metrics": metrics
            }
            
            logger.info("Complete JokePlanSearch pipeline finished successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {"error": str(e), "partial_results": self.get_performance_metrics()}

# Example usage and utility functions
def create_mock_api_client():
    """
    Create a mock API client for testing purposes.
    Replace this with your actual API client initialization.
    """
    class MockClient:
        def chat(self):
            return self
        
        def completions(self):
            return self
        
        def create(self, **kwargs):
            # Mock response - replace with actual API integration
            class MockResponse:
                def __init__(self):
                    self.choices = [self]
                    self.message = self
                    self.content = "This is a mock response for testing purposes."
                    self.usage = self
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

if __name__ == "__main__":
    # Example usage
    print("JokePlanSearch system initialized. Use create_mock_api_client() for testing.")
    print("For production use, replace with your actual LLM API client.") 