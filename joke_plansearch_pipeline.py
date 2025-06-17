from joke_plan_search_core import JokePlanSearch, BiasConfig
from joke_generation import JokeGenerationMixin
from joke_evaluation import JokeEvaluationMixin
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

class JokePlanSearchPipeline(JokePlanSearch, JokeGenerationMixin, JokeEvaluationMixin):
    """Pipeline orchestrator implementing the updated PlanSearch workflow."""

    def __init__(self, api_client=None, bias_config: Optional[BiasConfig] = None):
        super().__init__(api_client, bias_config)
        logger.info("Initialized JokePlanSearchPipeline with combinatorial PlanSearch")

    # ------------------------------------------------------------------
    # MAIN EXECUTION METHOD
    # ------------------------------------------------------------------
    def run_pipeline(self, topic: str):
        """Execute the complete PlanSearch joke generation pipeline for a topic."""
        start_time = time.time()
        print(f"🎯 Starting JokePlanSearch for topic: '{topic}'")

        # Step 1 – Topic analysis
        context_analysis = self.analyze_topic(topic).context_analysis

        # Step 2 – Primitive observations
        print("🧠 Generating primitive observations…")
        observations = self.generate_primitive_observations(topic, context_analysis)
        print(f"   • {len(observations)} observations generated")

        # Step 3 – Combinatorial plans
        print("🔄 Creating combinatorial plans…")
        plans = self.generate_combinatorial_plans(observations)
        print(f"   • {len(plans)} plans created")

        # Step 4 – Joke execution
        print("😂 Generating jokes from plans…")
        joke_objects = self.generate_full_jokes(topic, plans)

        # Step 5 – Refinement
        print("✨ Refining jokes…")
        refined_jokes = []
        for joke_data in joke_objects:
            refined = self.refine_joke(joke_data['joke'], topic, joke_data['plan']['plan_description'])
            joke_data['refined_joke'] = refined
            refined_jokes.append(joke_data)

        # Step 6 – Evaluation with bias analysis
        print("⚖️ Evaluating jokes…")
        evaluation_results = self.evaluate_jokes([j['refined_joke'] for j in refined_jokes], topic)

        # Step 7 – Present results
        output = self._present_results_with_plans(refined_jokes, evaluation_results)
        elapsed = time.time() - start_time
        print(f"\n✅ Pipeline finished in {elapsed:.2f}s – enjoy your laughs!")
        return output

    # ------------------------------------------------------------------
    # PRESENTATION UTILITIES
    # ------------------------------------------------------------------
    def _present_results_with_plans(self, jokes_with_plans, evaluation_results):
        """Pretty-print results including bias insights and plan origin."""
        # Pair jokes with their corresponding score dict
        scored = list(zip(jokes_with_plans, evaluation_results['scores']))
        scored.sort(key=lambda x: x[1].get('corrected_score', x[1]['overall_score']), reverse=True)

        bias = evaluation_results['bias_analysis']
        print("\n" + "=" * 80)
        print("🏆 JOKEPLANSEARCH RESULTS")
        print("=" * 80)
        print("\n📊 BIAS ANALYSIS:")
        print(f"   • First position advantage: {bias['first_position_advantage']:.2f}")
        print(f"   • Last position advantage: {bias['last_position_advantage']:.2f}")

        top_n = min(5, len(scored))
        print(f"\n🎭 TOP {top_n} JOKES (with combinatorial origins):")
        for idx, (joke_record, score_dict) in enumerate(scored[:top_n], 1):
            score_val = score_dict.get('corrected_score', score_dict['overall_score'])
            print(f"\n{idx}. SCORE: {score_val:.1f}/10")
            
            # Choose the best available joke text
            joke_to_display = self._select_best_joke_text(joke_record)
            print(f"   JOKE: {joke_to_display}")
            print(f"   PLAN: {joke_record['plan']['plan_description'][:100]}…")
            print(f"   OBSERVATIONS USED: {', '.join(joke_record['observations_used'])}")
            if 'bias_correction' in score_dict:
                print(f"   BIAS CORRECTION APPLIED: {score_dict['bias_correction']:+.2f}")
        return scored

    def _select_best_joke_text(self, joke_record):
        """Select the best joke text from available options."""
        refined_joke = joke_record.get('refined_joke', '')
        original_joke = joke_record.get('joke', '')
        
        # Check if refined joke looks like critique/meta-text
        critique_indicators = [
            'critique', 'analysis', 'suggestion', 'feedback', 'improvement', 
            'rationale', 'enhanced', 'original joke', 'improved joke',
            'punchline:', 'setup:', '**', 'the joke', 'version', 'explanation',
            'changes:', 'refinement', 'better', 'funnier', 'timing', 'impact'
        ]
        
        def looks_like_joke(text):
            """Check if text looks like an actual joke."""
            if not text or len(text) < 10:
                return False
            
            # Too long to be a joke
            if len(text) > 400:
                return False
            
            # Contains critique indicators
            if any(indicator in text.lower() for indicator in critique_indicators):
                return False
            
            # Starts with typical joke patterns
            joke_starters = ['why ', 'what ', 'how ', 'a ', 'an ', '"', 'i ', 'my ', 'the ']
            if any(text.lower().strip().startswith(starter) for starter in joke_starters):
                return True
            
            # Contains question marks (often in jokes)
            if '?' in text:
                return True
            
            # Reasonable length and no critique words
            return 20 < len(text) < 300
        
        # Check refined joke first
        if looks_like_joke(refined_joke):
            return refined_joke
        
        # Fall back to original joke
        if looks_like_joke(original_joke):
            return original_joke
        
        # If both contain critique text, try to extract the cleanest parts
        for joke_text in [refined_joke, original_joke]:
            if joke_text:
                lines = joke_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if looks_like_joke(line):
                        return line
        
        # Last resort: return the shorter of the two
        if refined_joke and original_joke:
            return refined_joke if len(refined_joke) < len(original_joke) else original_joke
        
        return refined_joke or original_joke or "No joke available"

# Quick utility for standalone execution
if __name__ == "__main__":
    pipeline = JokePlanSearchPipeline()
    try:
        topic_input = input("Enter a topic for joke generation: ").strip() or "programming"
    except EOFError:
        topic_input = "programming"
    pipeline.run_pipeline(topic_input) 