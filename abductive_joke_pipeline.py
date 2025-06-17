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

    def __init__(self, llm_client, bias_config: BiasConfig = None):
        """Initialize the abductive joke pipeline"""
        self.llm_client = llm_client
        self.bias_config = bias_config or BiasConfig()
        self.api_call_count = 0
        self.last_api_call = 0
        self.min_delay_between_calls = 1.0
        
        # Cache for premise generation to avoid redundant calls
        self.premise_cache = {}
        
        # Experimental tracking
        self.experiment_results = {}
        self.generated_jokes = []
        
    def call_llm(self, prompt: str, temperature: float = None) -> str:
        """Make an API call with rate limiting"""
        if temperature is None:
            temperature = self.bias_config.generation_temperature
            
        # Rate limiting
        time_since_last_call = time.time() - self.last_api_call
        if time_since_last_call < self.min_delay_between_calls:
            time.sleep(self.min_delay_between_calls - time_since_last_call)
        
        self.last_api_call = time.time()
        self.api_call_count += 1
        
        # Use the same LLM calling pattern as the existing pipeline
        if hasattr(self.llm_client, 'chat'):
            if hasattr(self.llm_client, '_api_key') or 'groq' in str(type(self.llm_client)).lower():
                # Groq API client
                response = self.llm_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            else:
                # OpenAI-style client
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1000
                )
                return response.choices[0].message.content
        elif hasattr(self.llm_client, 'messages'):
            # Anthropic-style client
            response = self.llm_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            raise RuntimeError(f"Unsupported LLM client type: {type(self.llm_client)}")
    
    def establish_joke_world_premises(self, topic: str) -> JokeWorld:
        """Generate grounding and absurd premises for a topic"""
        logger.info(f"Establishing joke world premises for topic: {topic}")
        
        # Check cache first
        if topic in self.premise_cache:
            logger.info("Using cached premises")
            return self.premise_cache[topic]
        
        prompt = self.PREMISE_GENERATION_PROMPT.format(topic=topic)
        response = self.call_llm(prompt, temperature=0.8)
        
        # Parse the response
        grounding_premise = None
        absurd_premise = None
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Grounding Premise:'):
                grounding_premise = JokePremise(
                    content=line[18:].strip(),
                    premise_type="grounding"
                )
            elif line.startswith('Absurd Premise:'):
                absurd_premise = JokePremise(
                    content=line[15:].strip(), 
                    premise_type="absurd"
                )
        
        if not grounding_premise or not absurd_premise:
            # Fallback parsing if format isn't exactly followed
            grounding_premise, absurd_premise = self._parse_premise_fallback(response, topic)
            if not grounding_premise or not absurd_premise:
                raise ValueError(f"Failed to parse premises from response: {response}")
        
        joke_world = JokeWorld(topic, grounding_premise, absurd_premise)
        
        # Cache the result
        self.premise_cache[topic] = joke_world
        
        logger.info(f"Generated premises - Grounding: {grounding_premise.content[:50]}...")
        logger.info(f"Generated premises - Absurd: {absurd_premise.content[:50]}...")
        
        return joke_world
    
    def _parse_premise_fallback(self, response: str, topic: str) -> Tuple[JokePremise, JokePremise]:
        """Fallback parsing method if structured format fails"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Look for any lines that might be premises
        potential_premises = []
        for line in lines:
            if len(line) > 20 and not line.startswith('Topic:') and not line.startswith('Requirements:'):
                potential_premises.append(line)
        
        if len(potential_premises) >= 2:
            grounding = JokePremise(potential_premises[0], "grounding")
            absurd = JokePremise(potential_premises[1], "absurd")
            return grounding, absurd
        
        # Ultimate fallback - generate basic premises
        grounding = JokePremise(f"{topic} is a common part of everyday life.", "grounding")
        absurd = JokePremise(f"{topic} secretly operates according to bizarre magical rules.", "absurd")
        return grounding, absurd
    
    def generate_abductive_joke(self, joke_world: JokeWorld) -> AbductiveJoke:
        """Generate complete joke with setup and punchline"""
        logger.info(f"Generating abductive joke for topic: {joke_world.topic}")
        
        prompt = self.ABDUCTIVE_JOKE_PROMPT.format(
            topic=joke_world.topic,
            grounding_premise=joke_world.grounding_premise.content,
            absurd_premise=joke_world.absurd_premise.content
        )
        
        response = self.call_llm(prompt, temperature=0.9)
        
        # Parse setup and punchline
        setup = ""
        punchline = ""
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Setup:'):
                setup = line[6:].strip()
            elif line.startswith('Punchline:'):
                punchline = line[10:].strip()
        
        if not setup or not punchline:
            # Fallback parsing
            setup, punchline = self._parse_joke_fallback(response)
        
        reasoning_chain = f"Grounding: {joke_world.grounding_premise.content} → Absurd: {joke_world.absurd_premise.content} → Abductive reasoning applied"
        
        joke = AbductiveJoke(
            joke_world=joke_world,
            setup=setup,
            punchline=punchline,
            reasoning_chain=reasoning_chain,
            metadata={
                'generation_temperature': 0.9,
                'api_calls_used': 1,
                'timestamp': time.time()
            }
        )
        
        logger.info(f"Generated joke: {setup[:30]}... → {punchline[:30]}...")
        return joke
    
    def _parse_joke_fallback(self, response: str) -> Tuple[str, str]:
        """Fallback parsing for joke response"""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) >= 2:
            setup = sentences[0] + '.'
            punchline = sentences[1] + '.'
            return setup, punchline
        
        # Ultimate fallback
        return "Here's an observation about this topic.", "And here's the surprising explanation."
    
    def generate_joke_batch(self, topic: str, num_jokes: int = 5) -> List[AbductiveJoke]:
        """Generate multiple jokes for comparison"""
        logger.info(f"Generating batch of {num_jokes} jokes for topic: {topic}")
        
        jokes = []
        
        # Generate multiple joke worlds for variety
        for i in range(num_jokes):
            try:
                # Create new premises each time for variety
                joke_world = self.establish_joke_world_premises(topic)
                joke = self.generate_abductive_joke(joke_world)
                jokes.append(joke)
                
                # Add to tracking
                self.generated_jokes.append(joke)
                
            except Exception as e:
                logger.error(f"Failed to generate joke {i+1}: {str(e)}")
                continue
        
        logger.info(f"Successfully generated {len(jokes)} jokes")
        return jokes


class AbductiveExperimentFramework:
    """Experimental design features for abductive joke research"""
    
    def __init__(self, pipeline: AbductiveJokePipeline, analyzer: JokeAnalyzer):
        self.pipeline = pipeline
        self.analyzer = analyzer
        self.experiment_results = {}
    
    def run_premise_type_experiment(self, topics: List[str], iterations_per_topic: int = 10) -> Dict[str, Any]:
        """
        Test hypothesis: Do jokes with one realistic + one absurd premise 
        score higher than jokes with two realistic premises?
        """
        logger.info(f"Running premise type experiment with {len(topics)} topics, {iterations_per_topic} iterations each")
        
        results = {
            'hypothesis': 'Realistic + Absurd premise combinations outperform realistic + realistic',
            'topics_tested': topics,
            'iterations_per_topic': iterations_per_topic,
            'results_by_topic': {},
            'aggregate_results': {}
        }
        
        for topic in topics:
            topic_results = {
                'abductive_jokes': [],
                'control_jokes': [],
                'abductive_scores': [],
                'control_scores': []
            }
            
            # Generate abductive jokes (realistic + absurd)
            for i in range(iterations_per_topic):
                try:
                    joke_world = self.pipeline.establish_joke_world_premises(topic)
                    joke = self.pipeline.generate_abductive_joke(joke_world)
                    consistency_score = self.analyzer.measure_logical_consistency(joke, joke_world)
                    
                    topic_results['abductive_jokes'].append(joke)
                    topic_results['abductive_scores'].append(consistency_score)
                    
                except Exception as e:
                    logger.error(f"Failed abductive generation for {topic}, iteration {i}: {e}")
            
            # For control group, we'd need to implement realistic + realistic premise generation
            # This is a placeholder for that comparison
            
            results['results_by_topic'][topic] = topic_results
        
        return results
    
    def run_abduction_effectiveness_experiment(self, topics: List[str]) -> Dict[str, Any]:
        """
        Test hypothesis: Do punchlines generated via abductive reasoning
        score higher than simpler generation methods?
        """
        logger.info(f"Running abduction effectiveness experiment with topics: {topics}")
        
        results = {
            'hypothesis': 'Abductive reasoning produces higher quality punchlines',
            'topics_tested': topics,
            'abductive_results': {},
            'baseline_results': {},
            'comparison_metrics': {}
        }
        
        for topic in topics:
            # Generate abductive jokes
            abductive_jokes = self.pipeline.generate_joke_batch(topic, num_jokes=5)
            
            # Analyze logical consistency
            consistency_scores = []
            for joke in abductive_jokes:
                score = self.analyzer.measure_logical_consistency(joke, joke.joke_world)
                consistency_scores.append(score)
            
            results['abductive_results'][topic] = {
                'jokes': [joke.to_dict() for joke in abductive_jokes],
                'consistency_scores': consistency_scores,
                'average_consistency': statistics.mean(consistency_scores) if consistency_scores else 0
            }
        
        return results
    
    def analyze_world_consistency_impact(self, jokes: List[AbductiveJoke]) -> Dict[str, Any]:
        """
        Measure correlation between logical consistency within the 
        established world and joke quality ratings
        """
        logger.info(f"Analyzing world consistency impact for {len(jokes)} jokes")
        
        consistency_scores = []
        for joke in jokes:
            score = self.analyzer.measure_logical_consistency(joke, joke.joke_world)
            consistency_scores.append(score)
        
        return {
            'total_jokes_analyzed': len(jokes),
            'consistency_scores': consistency_scores,
            'average_consistency': statistics.mean(consistency_scores) if consistency_scores else 0,
            'consistency_range': (min(consistency_scores), max(consistency_scores)) if consistency_scores else (0, 0),
            'standard_deviation': statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0
        }


class AbductivePlanSearchIntegration:
    """Integration points with existing PlanSearch infrastructure"""
    
    @staticmethod
    def integrate_with_plansearch(existing_pipeline, abductive_pipeline: AbductiveJokePipeline):
        """
        Allow side-by-side comparison between abductive method 
        and existing PlanSearch approach
        """
        return {
            'existing_pipeline': existing_pipeline,
            'abductive_pipeline': abductive_pipeline,
            'comparison_framework': 'Ready for A/B testing'
        }
    
    @staticmethod
    def export_for_evaluation(jokes: List[AbductiveJoke], format: str = "human_eval") -> Dict[str, Any]:
        """
        Export jokes in format suitable for human evaluation studies
        """
        if format == "human_eval":
            return {
                'jokes': [
                    {
                        'id': i,
                        'topic': joke.joke_world.topic,
                        'setup': joke.setup,
                        'punchline': joke.punchline,
                        'full_joke': joke.get_full_joke(),
                        'method': 'abductive_reasoning',
                        'grounding_premise': joke.joke_world.grounding_premise.content,
                        'absurd_premise': joke.joke_world.absurd_premise.content
                    }
                    for i, joke in enumerate(jokes)
                ],
                'evaluation_criteria': [
                    'humor_rating',
                    'logical_consistency',
                    'originality',
                    'relatability'
                ],
                'instructions': 'Rate each joke on a 1-10 scale for each criterion'
            }
        else:
            return {'error': f'Unsupported format: {format}'}


class AbductivePlanSearchIntegration:
    """Integration points with existing PlanSearch infrastructure"""
    
    @staticmethod
    def integrate_with_plansearch(existing_pipeline, abductive_pipeline: AbductiveJokePipeline):
        """
        Allow side-by-side comparison between abductive method 
        and existing PlanSearch approach
        """
        return {
            'existing_pipeline': existing_pipeline,
            'abductive_pipeline': abductive_pipeline,
            'comparison_framework': 'Ready for A/B testing'
        }
    
    @staticmethod
    def export_for_evaluation(jokes: List[AbductiveJoke], format: str = "human_eval") -> Dict[str, Any]:
        """
        Export jokes in format suitable for human evaluation studies
        """
        if format == "human_eval":
            return {
                'jokes': [
                    {
                        'id': i,
                        'topic': joke.joke_world.topic,
                        'setup': joke.setup,
                        'punchline': joke.punchline,
                        'full_joke': joke.get_full_joke(),
                        'method': 'abductive_reasoning',
                        'grounding_premise': joke.joke_world.grounding_premise.content,
                        'absurd_premise': joke.joke_world.absurd_premise.content
                    }
                    for i, joke in enumerate(jokes)
                ],
                'evaluation_criteria': [
                    'humor_rating',
                    'logical_consistency',
                    'originality',
                    'relatability'
                ],
                'instructions': 'Rate each joke on a 1-10 scale for each criterion'
            }
        else:
            return {'error': f'Unsupported format: {format}'}


def create_abductive_pipeline(llm_client) -> AbductiveJokePipeline:
    """Factory function to create a configured abductive pipeline"""
    return AbductiveJokePipeline(llm_client)


def run_abductive_demo(topic: str, llm_client, num_jokes: int = 3) -> Dict[str, Any]:
    """Run a quick demonstration of the abductive pipeline"""
    logger.info(f"Running abductive demo for topic: {topic}")
    
    pipeline = create_abductive_pipeline(llm_client)
    analyzer = JokeAnalyzer(llm_client)
    
    # Generate jokes
    jokes = pipeline.generate_joke_batch(topic, num_jokes)
    
    # Analyze results
    premise_analysis = analyzer.analyze_premise_types(jokes)
    
    # Calculate consistency scores
    consistency_scores = []
    for joke in jokes:
        score = analyzer.measure_logical_consistency(joke, joke.joke_world)
        consistency_scores.append(score)
    
    return {
        'topic': topic,
        'jokes_generated': len(jokes),
        'jokes': [joke.to_dict() for joke in jokes],
        'premise_analysis': premise_analysis,
        'consistency_scores': consistency_scores,
        'average_consistency': statistics.mean(consistency_scores) if consistency_scores else 0,
        'api_calls_used': pipeline.api_call_count + analyzer.api_call_count
    }



if __name__ == "__main__":
    # Example usage
    import os
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        import groq
        client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Run demo
        demo_results = run_abductive_demo("coffee shops", client, num_jokes=2)
        
        print("\n🎭 Abductive Joke Pipeline Demo Results")
        print("=" * 50)
        print(f"Topic: {demo_results['topic']}")
        print(f"Jokes Generated: {demo_results['jokes_generated']}")
        print(f"Average Consistency Score: {demo_results['average_consistency']:.2f}/10")
        print(f"API Calls Used: {demo_results['api_calls_used']}")
        
        for i, joke_data in enumerate(demo_results['jokes']):
            print(f"\nJoke {i+1}:")
            print(f"Setup: {joke_data['setup']}")
            print(f"Punchline: {joke_data['punchline']}")
            print(f"World - Grounding: {joke_data['joke_world']['grounding_premise']['content']}")
            print(f"World - Absurd: {joke_data['joke_world']['absurd_premise']['content']}")
            
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure GROQ_API_KEY is set and groq package is installed") 