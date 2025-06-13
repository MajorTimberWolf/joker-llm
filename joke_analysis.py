"""
JokePlanSearch: Final Analysis and Ranking (Phase 4)
Implements score aggregation, ranking, and comprehensive analysis generation.
"""

from joke_plan_search_core import JokePlanSearch, JokeCandidate
from dataclasses import asdict
import logging
import statistics
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class JokeAnalysisMixin:
    """Mixin class containing final analysis methods for JokePlanSearch."""
    
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
    
    def generate_detailed_report(self, include_bias_analysis: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive report with all results and analysis.
        
        Args:
            include_bias_analysis: Whether to include bias detection analysis
            
        Returns:
            Dictionary containing complete results report
        """
        logger.info("Generating detailed report")
        
        # Calculate final rankings
        ranked_jokes = self.calculate_final_rankings()
        
        # Generate analysis
        analysis = self.generate_final_analysis()
        
        # Get performance metrics
        metrics = self.get_performance_metrics()
        
        # Bias analysis if requested
        bias_analysis = {}
        if include_bias_analysis and hasattr(self, 'run_bias_detection_analysis'):
            try:
                bias_analysis = self.run_bias_detection_analysis()
            except Exception as e:
                logger.warning(f"Bias analysis failed: {str(e)}")
                bias_analysis = {"error": str(e)}
        
        # Compile comprehensive report
        report = {
            "experiment_summary": {
                "topic": self.topic_analysis.original_topic if self.topic_analysis else "Unknown",
                "total_jokes_generated": len(self.joke_candidates),
                "total_angles_explored": len(self.joke_angles),
                "evaluation_rounds_completed": len(set(
                    score_type for candidate in self.joke_candidates 
                    for score_type in candidate.scores.keys()
                )),
                "top_score": ranked_jokes[0][1] if ranked_jokes else 0,
                "average_score": statistics.mean([score for _, score in ranked_jokes]) if ranked_jokes else 0
            },
            
            "topic_analysis": asdict(self.topic_analysis) if self.topic_analysis else {},
            
            "ranked_jokes": [
                {
                    "rank": i + 1,
                    "joke": candidate.refined_joke or candidate.full_joke,
                    "angle": candidate.angle,
                    "final_score": score,
                    "confidence_interval": candidate.confidence_interval,
                    "detailed_scores": candidate.scores,
                    "comparative_record": {
                        "wins": candidate.comparative_wins,
                        "losses": candidate.comparative_losses,
                        "win_rate": candidate.comparative_wins / (candidate.comparative_wins + candidate.comparative_losses) 
                                  if (candidate.comparative_wins + candidate.comparative_losses) > 0 else 0
                    },
                    "refinement_history": {
                        "original_joke": candidate.full_joke,
                        "refined_joke": candidate.refined_joke,
                        "critique": candidate.critique,
                        "improvement_suggestions": candidate.improvement_suggestions
                    }
                }
                for i, (candidate, score) in enumerate(ranked_jokes)
            ],
            
            "analysis_insights": analysis,
            "performance_metrics": metrics,
            "bias_analysis": bias_analysis,
            
            "methodology_summary": {
                "generation_approach": "PlanSearch with multi-angle exploration",
                "evaluation_dimensions": ["cleverness", "surprise", "relatability", "timing", "overall"],
                "bias_mitigation_techniques": [
                    "Random joke ordering",
                    "Multiple evaluation rounds",
                    "Ensemble judging perspectives",
                    "Pairwise comparisons"
                ],
                "confidence_measures": "95% confidence intervals based on score variance"
            }
        }
        
        logger.info("Detailed report generated successfully")
        return report
    
    def export_results_summary(self, top_n: int = 10) -> str:
        """
        Generate a concise text summary of results for easy sharing.
        
        Args:
            top_n: Number of top jokes to include in summary
            
        Returns:
            Formatted text summary
        """
        ranked_jokes = self.calculate_final_rankings()
        
        if not ranked_jokes:
            return "No results to summarize."
        
        summary_lines = [
            f"JokePlanSearch Results for Topic: '{self.topic_analysis.original_topic}'",
            "=" * 60,
            f"Total jokes generated: {len(self.joke_candidates)}",
            f"Evaluation completed with {len(set(score_type for candidate in self.joke_candidates for score_type in candidate.scores.keys()))} different scoring dimensions",
            "",
            f"Top {min(top_n, len(ranked_jokes))} Jokes:",
            "-" * 40
        ]
        
        for i, (candidate, score) in enumerate(ranked_jokes[:top_n]):
            joke_text = candidate.refined_joke or candidate.full_joke
            confidence_low, confidence_high = candidate.confidence_interval
            
            summary_lines.extend([
                f"{i+1}. {joke_text}",
                f"   Score: {score:.2f} (95% CI: {confidence_low:.2f}-{confidence_high:.2f})",
                f"   Angle: {candidate.angle}",
                ""
            ])
        
        # Add performance summary
        metrics = self.get_performance_metrics()
        summary_lines.extend([
            "Performance Metrics:",
            "-" * 20,
            f"API calls made: {metrics['api_calls_made']}",
            f"Total tokens used: {metrics['total_tokens_used']}",
            f"Successful evaluations: {metrics['successful_evaluations']}"
        ])
        
        return "\n".join(summary_lines)
    
    def save_results_to_json(self, filepath: str, include_bias_analysis: bool = True) -> bool:
        """
        Save comprehensive results to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
            include_bias_analysis: Whether to include bias analysis
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            report = self.generate_detailed_report(include_bias_analysis)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results to {filepath}: {str(e)}")
            return False

# Extend the main class with analysis capabilities
class JokePlanSearchWithAnalysis(JokePlanSearch, JokeAnalysisMixin):
    """JokePlanSearch with analysis capabilities."""
    pass 