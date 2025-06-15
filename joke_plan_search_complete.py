"""
JokePlanSearch: Complete Integrated System
A comprehensive LLM-based joke generation and evaluation system.

This module combines all components to provide the complete PlanSearch methodology
implementation for computational humor generation with bias-minimized evaluation.
"""

from joke_plan_search_core import JokePlanSearch, BiasConfig, JokeCandidate, TopicAnalysis
from joke_generation import JokeGenerationMixin
from joke_evaluation import JokeEvaluationMixin
from joke_analysis import JokeAnalysisMixin
from dataclasses import asdict
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class JokePlanSearchComplete(
    JokePlanSearch, 
    JokeGenerationMixin, 
    JokeEvaluationMixin, 
    JokeAnalysisMixin
):
    """
    Complete JokePlanSearch system implementing the full pipeline:
    - Phase 1: Setup and Configuration
    - Phase 2: Topic Analysis and Joke Generation
    - Phase 3: Multi-dimensional Evaluation with Bias Mitigation
    - Phase 4: Final Analysis and Ranking
    """
    
    def __init__(self, api_client=None, bias_config: BiasConfig = None):
        """
        Initialize the complete JokePlanSearch system.
        
        Args:
            api_client: Initialized LLM API client (OpenAI, Anthropic, Groq, etc.)
            bias_config: Configuration for bias mitigation techniques
        """
        super().__init__(api_client, bias_config)
        logger.info("JokePlanSearch Complete system initialized")
    
    def run_complete_pipeline(self, topic: str, **kwargs) -> Dict[str, Any]:
        """
        Run the complete JokePlanSearch pipeline from topic to final analysis.
        
        Args:
            topic: The input topic for joke generation
            **kwargs: Additional configuration options:
                - refinement_rounds (int): Number of joke refinement rounds (default: 2)
                - top_n (int): Number of top jokes to analyze in detail (default: 5)
                - min_comparisons (int): Minimum pairwise comparisons per joke
                - similarity_threshold (float): Threshold for joke similarity detection
                - include_bias_analysis (bool): Whether to include bias detection
                - num_angles (int): Number of joke angles to generate (default: 11)
                - max_jokes (int): Maximum number of jokes to generate (default: 11)
                
        Returns:
            Dictionary containing complete results and analysis
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting complete JokePlanSearch pipeline for topic: '{topic}'")
            
            # Apply free tier optimizations if specified
            num_angles = kwargs.get('num_angles', 11)
            max_jokes = kwargs.get('max_jokes', 11)
            
            # Set expected API calls for rate limiting optimization
            estimated_calls = 5 + num_angles + max_jokes * 2 + kwargs.get('refinement_rounds', 2) * max_jokes + 10
            self.expected_api_calls = estimated_calls
            
            # Phase 1: Setup (already done in __init__)
            logger.info("Phase 1: System setup completed")
            
            # Phase 2: Topic Analysis and Joke Generation
            logger.info("Phase 2: Beginning joke generation pipeline")
            
            # Step 2.1: Topic Analysis
            self.analyze_topic(topic)
            
            # Step 2.2: Generate Diverse Joke Angles (with optimization)
            if hasattr(self, 'generate_diverse_joke_angles'):
                # Store original method temporarily
                original_method = self.generate_diverse_joke_angles
                
                # Override for optimized generation
                def optimized_angles():
                    angles = original_method()
                    return angles[:num_angles]  # Limit angles for free tier
                
                self.generate_diverse_joke_angles = optimized_angles
                
            self.generate_diverse_joke_angles()
            
            # Step 2.3: Generate Jokes from Angles (with optimization)
            self.generate_jokes_from_angles()
            
            # Limit jokes for free tier optimization
            if len(self.joke_candidates) > max_jokes:
                self.joke_candidates = self.joke_candidates[:max_jokes]
            
            # Step 2.4: Refine Jokes
            refinement_rounds = kwargs.get('refinement_rounds', 2)
            if refinement_rounds > 0:
                self.refine_jokes(refinement_rounds)
            
            # Step 2.5: Ensure Diversity
            similarity_threshold = kwargs.get('similarity_threshold', 0.7)
            self.ensure_joke_diversity(similarity_threshold)
            
            logger.info(f"Phase 2 completed: Generated {len(self.joke_candidates)} jokes")
            
            # Phase 3: Evaluation
            logger.info("Phase 3: Beginning evaluation pipeline")
            
            # Step 3.2: Multi-dimensional Evaluation
            eval_result_1 = self.evaluate_jokes_multidimensional()
            
            # Step 3.3: Comparative Evaluation
            min_comparisons = kwargs.get('min_comparisons', None)
            eval_result_2 = self.evaluate_jokes_comparative(min_comparisons)
            
            # Step 3.4: Ensemble Evaluation
            eval_result_3 = self.evaluate_jokes_ensemble()
            
            logger.info("Phase 3 completed: All evaluation rounds finished")
            
            # Phase 4: Final Analysis
            logger.info("Phase 4: Beginning final analysis")
            
            # Step 4.1: Calculate Rankings
            ranked_jokes = self.calculate_final_rankings()
            
            # Step 4.2: Generate Analysis
            top_n = kwargs.get('top_n', 5)
            analysis = self.generate_final_analysis(top_n)
            
            # Step 4.3: Compile Results
            include_bias_analysis = kwargs.get('include_bias_analysis', True)
            detailed_report = self.generate_detailed_report(include_bias_analysis)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Compile final results
            results = {
                "success": True,
                "execution_time_seconds": execution_time,
                "pipeline_results": detailed_report,
                "evaluation_summary": {
                    "multidimensional": eval_result_1,
                    "comparative": eval_result_2,
                    "ensemble": eval_result_3
                },
                "quick_summary": self.export_results_summary(top_n)
            }
            
            logger.info(f"Complete JokePlanSearch pipeline finished successfully in {execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Pipeline failed after {execution_time:.2f} seconds: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time_seconds": execution_time,
                "partial_results": self.get_performance_metrics()
            }
    
    def run_quick_demo(self, topic: str) -> str:
        """
        Run a simplified version of the pipeline for quick demonstrations.
        
        Args:
            topic: The input topic for joke generation
            
        Returns:
            Formatted string with top jokes and basic analysis
        """
        try:
            # Simplified pipeline with reduced parameters for speed
            results = self.run_complete_pipeline(
                topic,
                refinement_rounds=1,
                top_n=3,
                include_bias_analysis=False
            )
            
            if results["success"]:
                return results["quick_summary"]
            else:
                return f"Demo failed: {results.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"Demo failed: {str(e)}"
    
    def batch_process_topics(self, topics: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple topics in batch mode.
        
        Args:
            topics: List of topics to process
            **kwargs: Configuration options passed to run_complete_pipeline
            
        Returns:
            Dictionary mapping topics to their results
        """
        logger.info(f"Starting batch processing of {len(topics)} topics")
        
        batch_results = {}
        
        for i, topic in enumerate(topics):
            logger.info(f"Processing topic {i+1}/{len(topics)}: '{topic}'")
            
            try:
                # Reset state for new topic
                self.topic_analysis = None
                self.joke_angles = []
                self.joke_candidates = []
                self.evaluation_results = {}
                
                # Process topic
                results = self.run_complete_pipeline(topic, **kwargs)
                batch_results[topic] = results
                
                logger.info(f"Completed topic '{topic}' successfully")
                
            except Exception as e:
                logger.error(f"Failed to process topic '{topic}': {str(e)}")
                batch_results[topic] = {
                    "success": False,
                    "error": str(e)
                }
        
        logger.info(f"Batch processing completed for {len(topics)} topics")
        return batch_results
    
    def export_batch_results(self, batch_results: Dict[str, Dict[str, Any]], 
                           output_format: str = "json") -> str:
        """
        Export batch processing results to various formats.
        
        Args:
            batch_results: Results from batch_process_topics
            output_format: Format for export ("json", "csv", "txt")
            
        Returns:
            Formatted string with results
        """
        if output_format == "json":
            import json
            return json.dumps(batch_results, indent=2, ensure_ascii=False)
        
        elif output_format == "csv":
            # Create CSV with top joke from each topic
            csv_lines = ["Topic,Top_Joke,Score,Execution_Time"]
            
            for topic, results in batch_results.items():
                if results.get("success") and "pipeline_results" in results:
                    ranked_jokes = results["pipeline_results"].get("ranked_jokes", [])
                    if ranked_jokes:
                        top_joke = ranked_jokes[0]
                        csv_lines.append(f'"{topic}","{top_joke["joke"]}",{top_joke["final_score"]},{results["execution_time_seconds"]}')
                    else:
                        csv_lines.append(f'"{topic}","No jokes generated",0,{results.get("execution_time_seconds", 0)}')
                else:
                    csv_lines.append(f'"{topic}","Error: {results.get("error", "Unknown")}",0,{results.get("execution_time_seconds", 0)}')
            
            return "\n".join(csv_lines)
        
        elif output_format == "txt":
            # Create readable text summary
            txt_lines = ["JokePlanSearch Batch Results", "=" * 50, ""]
            
            for topic, results in batch_results.items():
                txt_lines.append(f"Topic: {topic}")
                txt_lines.append("-" * len(f"Topic: {topic}"))
                
                if results.get("success"):
                    summary = results.get("quick_summary", "No summary available")
                    txt_lines.append(summary)
                else:
                    txt_lines.append(f"Error: {results.get('error', 'Unknown error')}")
                
                txt_lines.extend(["", ""])  # Add spacing
            
            return "\n".join(txt_lines)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

# Utility functions for easy setup and usage

def create_openai_client(api_key: str):
    """
    Create an OpenAI client for use with JokePlanSearch.
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        Configured OpenAI client
    """
    try:
        import openai
        return openai.OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("OpenAI library not installed. Run: pip install openai")

def create_anthropic_client(api_key: str):
    """
    Create an Anthropic client for use with JokePlanSearch.
    
    Args:
        api_key: Anthropic API key
        
    Returns:
        Configured Anthropic client
    """
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError("Anthropic library not installed. Run: pip install anthropic")

def create_groq_client(api_key: str):
    """
    Create a Groq client for use with JokePlanSearch.
    
    Args:
        api_key: Groq API key
        
    Returns:
        Configured Groq client
    """
    try:
        import groq
        return groq.Groq(api_key=api_key)
    except ImportError:
        raise ImportError("Groq library not installed. Run: pip install groq")

def setup_for_colab():
    """
    Setup function optimized for Google Colab environment.
    
    Returns:
        Configuration dictionary for Colab usage
    """
    # Configure logging for Colab
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Return Colab-optimized configuration
    return {
        "bias_config": BiasConfig(),
        "default_kwargs": {
            "refinement_rounds": 2,
            "top_n": 5,
            "include_bias_analysis": True
        }
    }

# Example usage patterns

def example_usage():
    """
    Example usage patterns for JokePlanSearch.
    """
    examples = """
    # Basic Usage
    from joke_plan_search_complete import JokePlanSearchComplete, create_openai_client
    
    # Setup
    api_client = create_openai_client("your-api-key")
    joke_search = JokePlanSearchComplete(api_client)
    
    # Run pipeline
    results = joke_search.run_complete_pipeline("artificial intelligence")
    
    # Get summary
    print(results["quick_summary"])
    
    # Quick demo
    summary = joke_search.run_quick_demo("coffee")
    print(summary)
    
    # Batch processing
    topics = ["cats", "programming", "cooking"]
    batch_results = joke_search.batch_process_topics(topics)
    
    # Export results
    csv_output = joke_search.export_batch_results(batch_results, "csv")
    """
    return examples

if __name__ == "__main__":
    print("JokePlanSearch Complete System")
    print("=" * 40)
    print("A comprehensive LLM-based joke generation and evaluation system.")
    print("\nFor usage examples, call example_usage()")
    print("For Colab setup, call setup_for_colab()")
    print("\nReady to generate and evaluate jokes using the PlanSearch methodology!") 