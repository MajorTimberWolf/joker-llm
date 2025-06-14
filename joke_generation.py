"""
JokePlanSearch: Joke Generation Pipeline (Phase 2)
Implements topic analysis, diverse angle generation, and structured joke creation.
"""

from joke_plan_search_core import JokePlanSearch, JokeCandidate, TopicAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import random
import statistics

logger = logging.getLogger(__name__)

class JokeGenerationMixin:
    """Mixin class containing joke generation methods for JokePlanSearch."""
    
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
    
    def generate_diverse_joke_angles(self) -> list:
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
    
    def _parse_numbered_list(self, response: str) -> list:
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
    
    def _deduplicate_angles(self, angles: list) -> list:
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
    
    def generate_jokes_from_angles(self) -> list:
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
    
    def _parse_joke_response(self, response: str) -> tuple:
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
    
    def _parse_critique_response(self, response: str) -> tuple:
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
    
    def generate_primitive_observations(self, topic: str, context_analysis: str, num_observations: int = 8) -> list:
        """Generate primitive observations that will be combined into plans."""
        prompt = f"""
Generate {num_observations} distinct, primitive observations about \"{topic}\" that could be building blocks for jokes.

Each observation should be:
- A single, focused insight or characteristic
- Combinable with other observations
- Not a complete joke concept, but a building block

Context: {context_analysis}

Format as simple statements:
1. [Single primitive observation]
2. [Single primitive observation]
3. [Single primitive observation]
4. [Single primitive observation]
5. [Single primitive observation]
6. [Single primitive observation]
7. [Single primitive observation]
8. [Single primitive observation]
"""

        response = self.call_llm(prompt)
        observations = self._parse_numbered_list(response)
        logger.info(f"Generated {len(observations)} primitive observations")
        return observations

    def generate_combinatorial_plans(self, observations: list, max_plans: int = 15) -> list:
        """Generate joke plans by combining 2-4 primitive observations."""
        import itertools
        import random

        plans = []
        # Ensure reproducibility can be turned off/on by user; default pseudo-random
        random.seed()  # do not set a fixed seed

        # Split the plan budget roughly equally among combo sizes
        budget_per_size = max(1, max_plans // 3)

        for combo_size in [2, 3, 4]:
            combinations = list(itertools.combinations(observations, combo_size))
            if not combinations:
                continue
            sampled_combos = random.sample(combinations, min(len(combinations), budget_per_size))
            for combo in sampled_combos:
                plan_prompt = f"""
Create a coherent joke plan by combining these observations:
{chr(10).join(f"- {obs}" for obs in combo)}

Your task:
1. Find a unifying theme or connection between these observations
2. Describe how they could work together in a joke structure
3. Specify the joke format (one-liner, setup-punchline, dialogue, etc.)

Plan Description: [How these observations combine into a joke strategy]
Joke Format: [Specific format this plan will use]
"""
                response = self.call_llm(plan_prompt)
                plans.append({
                    'observations': combo,
                    'plan_description': response.strip(),
                    'combo_size': combo_size
                })
                if len(plans) >= max_plans:
                    break
            if len(plans) >= max_plans:
                break

        logger.info(f"Created {len(plans)} combinatorial plans")
        return plans[:max_plans]

    def generate_full_jokes(self, topic: str, plans: list) -> list:
        """Generate jokes from combinatorial plans."""
        joke_results = []
        for plan in plans:
            joke_prompt = f"""
Topic: \"{topic}\"
Plan: {plan['plan_description']}
Source Observations: {', '.join(plan['observations'])}

Execute this plan to create a complete joke. Follow the plan's structure and format exactly.

Joke: [Your complete joke here]
"""
            response = self.call_llm(joke_prompt)
            joke_results.append({
                'joke': response.strip(),
                'plan': plan,
                'observations_used': plan['observations']
            })
        logger.info(f"Generated {len(joke_results)} jokes from combinatorial plans")
        return joke_results

    def refine_joke(self, joke: str, topic: str, plan_description: str) -> str:
        """Refine a single joke using critique-and-revise loop."""
        critique_prompt = f"""Analyze this joke for improvement opportunities given the plan context.
Topic: \"{topic}\"
Plan: {plan_description}
Joke: \"{joke}\"

Provide specific feedback on:
1. Setup clarity
2. Punchline impact
3. Timing and brevity
4. Relevance to the topic \"{topic}\"

Critique: [detailed feedback]
Improvement Suggestions: [actionable changes]
"""
        critique_response = self.call_llm(critique_prompt)
        critique, suggestions = self._parse_critique_response(critique_response)

        refinement_prompt = f"""Original Joke: \"{joke}\"
Critique: {critique}
Improvement Suggestions: {suggestions}

Write an improved version of the joke that addresses every suggestion while preserving the core concept.

Improved Joke: [your refined joke]
"""
        refined_response = self.call_llm(refinement_prompt)
        refined_joke = self._extract_refined_joke(refined_response)
        return refined_joke.strip()

# Extend the main class with generation capabilities
class JokePlanSearchWithGeneration(JokePlanSearch, JokeGenerationMixin):
    """JokePlanSearch with joke generation capabilities."""
    pass 