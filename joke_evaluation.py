"""
JokePlanSearch: Joke Evaluation Pipeline (Phase 3)
Implements LLM-as-a-judge evaluation with bias mitigation techniques.
"""

from joke_plan_search_core import JokePlanSearch
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import random
import statistics
from typing import Dict, List, Any
import numpy as np
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

class JokeEvaluationMixin:
    """Mixin class containing joke evaluation methods for JokePlanSearch."""
    
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
    
    def run_bias_detection_analysis(self) -> Dict[str, Any]:
        """
        Additional method: Analyze evaluation results for systematic biases.
        
        Returns:
            Dictionary containing bias analysis results
        """
        if not self.joke_candidates:
            return {"error": "No jokes to analyze"}
        
        logger.info("Running bias detection analysis")
        
        # Analyze score distributions
        all_scores = []
        score_categories = {}
        
        for candidate in self.joke_candidates:
            for score_type, score in candidate.scores.items():
                if score_type not in score_categories:
                    score_categories[score_type] = []
                score_categories[score_type].append(score)
                all_scores.append(score)
        
        # Calculate statistics
        bias_analysis = {
            "total_evaluations": len(all_scores),
            "overall_mean": statistics.mean(all_scores) if all_scores else 0,
            "overall_std": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
            "score_type_analysis": {}
        }
        
        # Analyze each score type
        for score_type, scores in score_categories.items():
            if scores:
                bias_analysis["score_type_analysis"][score_type] = {
                    "mean": statistics.mean(scores),
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        # Check for position bias in comparative evaluations
        position_bias_info = self._analyze_position_bias(self.joke_candidates, self.scores_by_round)
        bias_analysis["position_bias"] = position_bias_info
        
        logger.info("Bias detection analysis completed")
        return bias_analysis
    
    def _analyze_position_bias(self, jokes, scores_by_round):
        """Calculate actual positional bias in LLM judgments."""
        bias_analysis = {
            'position_correlation': {},
            'first_position_advantage': 0,
            'last_position_advantage': 0,
            'statistical_significance': {}
        }

        for round_name, round_data in scores_by_round.items():
            if not round_data:
                continue
            # Ensure aligned ordering
            positions = list(range(len(round_data)))
            scores = [item['overall_score'] for item in round_data]
            if len(set(scores)) < 2:  # correlation undefined for constant lists
                correlation, p_value = 0, 1
            else:
                correlation, p_value = stats.pearsonr(positions, scores)
            bias_analysis['position_correlation'][round_name] = {
                'correlation': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            # First vs middle vs last comparisons
            first_score = round_data[0]['overall_score']
            last_score = round_data[-1]['overall_score']
            middle_scores = [d['overall_score'] for d in round_data[1:-1]] if len(round_data) > 2 else []

            if middle_scores:
                bias_analysis['first_position_advantage'] += first_score - np.mean(middle_scores)
                bias_analysis['last_position_advantage'] += last_score - np.mean(middle_scores)

        # Average over rounds
        num_rounds = max(1, len(scores_by_round))
        bias_analysis['first_position_advantage'] /= num_rounds
        bias_analysis['last_position_advantage'] /= num_rounds
        return bias_analysis

    def _apply_bias_correction(self, scores, bias_analysis):
        """Apply corrections based on detected positional bias."""
        corrected_scores = []
        first_adv = bias_analysis.get('first_position_advantage', 0)
        last_adv = bias_analysis.get('last_position_advantage', 0)

        for i, score_data in enumerate(scores):
            correction = 0
            if i == 0 and first_adv > 0.5:  # threshold for applying correction
                correction -= first_adv * 0.5
            if i == len(scores) - 1 and last_adv > 0.5:
                correction -= last_adv * 0.5
            corrected_score = max(1, min(10, score_data['overall_score'] + correction))
            corrected_scores.append({
                **score_data,
                'corrected_score': corrected_score,
                'bias_correction': correction
            })
        return corrected_scores

    def _calculate_consensus_scores(self, jokes, scores_by_round):
        """Aggregate scores across rounds into consensus list."""
        score_accumulator = defaultdict(list)
        for round_data in scores_by_round.values():
            for idx, data in enumerate(round_data):
                score_accumulator[idx].append(data['overall_score'])
        consensus = []
        for idx in range(len(jokes)):
            if score_accumulator[idx]:  # Check if list is not empty
                avg_score = sum(score_accumulator[idx]) / len(score_accumulator[idx])
            else:
                avg_score = 5.0  # Default fallback score
            consensus.append({
                'overall_score': avg_score
            })
        return consensus

    def _evaluate_round(self, ordered_jokes: list, round_name: str):
        """Evaluate a list of jokes and return structured scores."""
        prompt = f"""
Evaluate the following jokes about the same topic. Provide a holistic funniness score (1-10) for each and a one-line justification.

{chr(10).join(f"{i+1}. {j}" for i, j in enumerate(ordered_jokes))}

Return format:
JOKE [number]: [score] - [justification]
"""
        response = self.call_llm_judge(prompt)
        parsed = self._parse_simple_scores_with_reasons(response)
        return parsed

    def _parse_simple_scores_with_reasons(self, response: str):
        lines = response.split('\n')
        results = []
        for line in lines:
            if line.startswith('JOKE') and ':' in line:
                try:
                    parts = line.split(':', 1)[1].strip()
                    score_part, *reason_parts = parts.split('-', 1)
                    score = float(''.join(c for c in score_part if c.isdigit() or c == '.'))
                    reason = reason_parts[0].strip() if reason_parts else ''
                    results.append({'overall_score': score, 'reason': reason})
                except ValueError:
                    continue
        return results

    def evaluate_jokes(self, jokes: list, topic: str):
        """Evaluate jokes with proper bias detection and correction."""
        print("🔍 Running multi-round evaluation with bias analysis...")
        all_rounds = {}
        for round_num in range(3):
            shuffled = jokes.copy()
            random.shuffle(shuffled)
            round_name = f"round_{round_num+1}"
            all_rounds[round_name] = self._evaluate_round(shuffled, round_name)
        # Store for optional external analysis
        self.scores_by_round = all_rounds
        bias_analysis = self._analyze_position_bias(jokes, all_rounds)
        consensus_scores = self._calculate_consensus_scores(jokes, all_rounds)
        corrected_scores = self._apply_bias_correction(consensus_scores, bias_analysis)
        return {
            'scores': corrected_scores,
            'bias_analysis': bias_analysis,
            'round_details': all_rounds,
            'methodology': 'multi_round_with_bias_correction'
        }

# Extend the main class with evaluation capabilities
class JokePlanSearchWithEvaluation(JokePlanSearch, JokeEvaluationMixin):
    """JokePlanSearch with joke evaluation capabilities."""
    pass 