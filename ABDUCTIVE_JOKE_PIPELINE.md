# Abductive Joke Pipeline

## Overview

The **Abductive Joke Pipeline** is a novel approach to computational humor that generates jokes using formal reasoning principles instead of traditional brainstorming. This system creates jokes by establishing logical "worlds" with specific premises and then using abductive reasoning to generate surprising but internally consistent punchlines.

## Core Concept

The pipeline mimics sophisticated joke construction through a three-step process:

1. **Establish World Rules**: Define both normal and absurd premises
2. **Present Puzzling Observation**: Create a scenario that needs explanation  
3. **Abductive Leap**: Provide the most surprising explanation that's still logically consistent within the established world

## Key Features

### 🧠 **Formal Reasoning Approach**
- Uses logical premise structures (grounding + absurd)
- Applies abductive reasoning for punchline generation
- Maintains internal consistency within joke worlds

### 🔬 **Research-Grade Framework**
- Systematic experimental design capabilities
- Built-in evaluation and analysis tools
- Statistical significance testing support

### 📊 **Comprehensive Analytics**
- Logical consistency scoring
- Premise type analysis
- Comparative studies against traditional methods

### 🔗 **Integration Ready**
- Compatible with existing PlanSearch infrastructure
- Export functionality for human evaluation studies
- A/B testing framework included

## Architecture

### Core Classes

```python
class JokePremise:
    """Represents a single premise in the joke world"""
    content: str
    premise_type: str  # "grounding" or "absurd"

class JokeWorld:
    """Represents the complete world context for a joke"""
    topic: str
    grounding_premise: JokePremise
    absurd_premise: JokePremise

class AbductiveJoke:
    """Complete joke with reasoning chain and metadata"""
    joke_world: JokeWorld
    setup: str
    punchline: str
    reasoning_chain: str
    metadata: Dict[str, Any]
```

### Pipeline Components

1. **AbductiveJokePipeline**: Main generation engine
2. **JokeAnalyzer**: Research and evaluation tools
3. **AbductiveExperimentFramework**: Experimental design
4. **AbductivePlanSearchIntegration**: Compatibility layer

## Usage Examples

### Basic Generation

```python
from abductive_joke_pipeline import AbductiveJokePipeline
import groq

# Setup
client = groq.Groq(api_key="your-api-key")
pipeline = AbductiveJokePipeline(client)

# Generate joke world
joke_world = pipeline.establish_joke_world_premises("coffee shops")
print(f"Grounding: {joke_world.grounding_premise.content}")
print(f"Absurd: {joke_world.absurd_premise.content}")

# Generate joke
joke = pipeline.generate_abductive_joke(joke_world)
print(f"Setup: {joke.setup}")
print(f"Punchline: {joke.punchline}")
```

### Batch Generation

```python
# Generate multiple jokes for comparison
jokes = pipeline.generate_joke_batch("restaurants", num_jokes=5)

for i, joke in enumerate(jokes):
    print(f"Joke {i+1}: {joke.get_full_joke()}")
```

### Research Analysis

```python
from abductive_joke_pipeline import JokeAnalyzer

analyzer = JokeAnalyzer(client)

# Analyze premise patterns
premise_analysis = analyzer.analyze_premise_types(jokes)
print(f"Grounding themes: {premise_analysis['grounding_themes']}")
print(f"Absurd themes: {premise_analysis['absurd_themes']}")

# Measure logical consistency
for joke in jokes:
    score = analyzer.measure_logical_consistency(joke, joke.joke_world)
    print(f"Consistency: {score:.1f}/10")
```

### Experimental Framework

```python
from abductive_joke_pipeline import AbductiveExperimentFramework

experiment = AbductiveExperimentFramework(pipeline, analyzer)

# Run premise type experiment
results = experiment.run_premise_type_experiment(
    topics=["movies", "exercise"],
    iterations_per_topic=5
)

# Run effectiveness experiment  
effectiveness = experiment.run_abduction_effectiveness_experiment(
    topics=["technology", "cooking"]
)
```

## Demo Scripts

### Quick Demo
```bash
python run_abductive_demo.py basic
```

### Specific Components
```bash
python run_abductive_demo.py premises      # Premise analysis
python run_abductive_demo.py consistency   # Logical consistency
python run_abductive_demo.py experiment    # Research framework
python run_abductive_demo.py export        # Human evaluation export
python run_abductive_demo.py integration   # PlanSearch integration
```

### Comprehensive Demo
```bash
python run_abductive_demo.py  # Runs all components
```

## Example Output

### Generated Joke World
```
Topic: coffee shops
Grounding Premise: Coffee shops serve caffeinated beverages and often have cozy atmospheres where people go to relax, work, or socialize.

Absurd Premise: In every coffee shop, there exists a hidden "Caffeine King" espresso machine that, once a day, randomly selects a customer's cup and transforms it into a magical elixir that temporarily grants the drinker the ability to speak any language fluently, but only to order more coffee.
```

### Generated Joke
```
Setup: Regularly, at a popular local coffee shop, a seemingly monolingual tourist would walk in, order a simple coffee in broken English, and then, to everyone's surprise, begin conversing fluently with the barista and other patrons in perfect, idiomatic English - but only to discuss coffee blends, roast levels, and brewing methods.

Punchline: It's because the Caffeine King espresso machine had been selecting this tourist's cup every day, granting them the magical ability to speak any language fluently, but only to order more coffee, which the tourist, being very particular about their coffee, had been doing with increasing complexity and sophistication.

Logical Consistency Score: 8.0/10
```

## Prompt Engineering

### Premise Generation Prompt
The system uses carefully crafted prompts that enforce the grounding + absurd structure:

```
You are an expert in comedy and logic. For the given topic, your task is to establish a "joke world" by defining a set of premises or rules. You must provide two types of premises:

1. **Grounding Premise:** A true, normal, or stereotypical fact about the topic...
2. **Absurd Premise:** A completely unexpected or surreal rule...

[Detailed requirements and examples follow]
```

### Abductive Reasoning Prompt
The joke generation uses explicit abductive reasoning structure:

```
Your task is to create a complete joke based on a pre-defined "world" using abductive reasoning principles...

ABDUCTIVE REASONING STRUCTURE:
1. Setup: Present a situation or observation that seems puzzling
2. Implicit Question: Make the audience wonder "why?" or "how?"  
3. Punchline: Provide surprising explanation that makes sense given the absurd premise

[Detailed requirements follow]
```

## Research Applications

### Hypothesis Testing
The framework enables testing specific hypotheses about joke construction:

1. **Premise Type Effectiveness**: Do grounding + absurd combinations outperform other structures?
2. **Logical Consistency Impact**: Does internal consistency correlate with humor ratings?
3. **Abductive vs Traditional**: How does formal reasoning compare to brainstorming?

### Evaluation Metrics
- **Logical Consistency**: 1-10 scale measuring premise adherence
- **Premise Theme Analysis**: Categorization of successful patterns
- **Comparative Performance**: Statistical comparison with baselines

### Export for Human Studies
```python
export_data = AbductivePlanSearchIntegration.export_for_evaluation(
    jokes, format="human_eval"
)
# Produces structured data for human evaluation platforms
```

## Technical Implementation

### Error Handling
- Robust fallback parsing for LLM responses
- Rate limiting and API error management
- Quality validation for premise generation

### Performance Optimization
- Premise caching to avoid redundant calls
- Parallel processing for batch operations
- Configurable temperature settings for creativity vs consistency

### Integration Points
- Compatible with existing Groq/OpenAI/Anthropic clients
- Plugs into PlanSearch evaluation framework
- Exports to standard research data formats

## Configuration

### Groq Integration
The pipeline integrates with the existing Groq configuration system:

```python
from groq_config import get_recommended_config

config = get_recommended_config("balanced")
# Automatically selects optimal model and settings
```

### Bias Configuration
Uses the existing BiasConfig system for evaluation parameters:

```python
from joke_plan_search_core import BiasConfig

bias_config = BiasConfig()
bias_config.evaluation_rounds = 3
bias_config.generation_temperature = 0.8

pipeline = AbductiveJokePipeline(client, bias_config)
```

## Research Value

### Novel Contribution
- First implementation of formal abductive reasoning for humor
- Systematic approach to joke world construction
- Measurable logical consistency in humor generation

### Scientific Rigor
- Reproducible experimental framework
- Statistical significance testing
- Controlled variables and proper baselines

### Practical Applications
- Improved computational humor quality
- Better understanding of joke mechanics
- Framework for humor theory research

## File Structure

```
joker-llm/
├── abductive_joke_pipeline.py     # Core implementation
├── run_abductive_demo.py          # Demo and testing script
├── ABDUCTIVE_JOKE_PIPELINE.md     # This documentation
├── groq_config.py                 # Integration with existing config
└── joke_plan_search_core.py       # Base classes and utilities
```

## Dependencies

- groq (for API access)
- joke_plan_search_core (existing pipeline classes)
- Standard Python libraries: json, time, logging, statistics, typing

## Future Extensions

### Planned Features
1. **Multi-Premise Worlds**: Support for 3+ premise combinations
2. **Premise Refinement**: Iterative improvement of world rules
3. **Narrative Jokes**: Extended multi-setup joke structures
4. **Cultural Context**: Premise adaptation for different audiences

### Research Directions
1. **Cognitive Modeling**: Alignment with human reasoning patterns
2. **Creativity Metrics**: Measuring novelty and surprise quantitatively
3. **Domain Adaptation**: Specialized premise types for different topics
4. **Interactive Generation**: User-guided premise and reasoning selection

## Citation

If you use this implementation in research, please cite:

```
Abductive Joke Pipeline: A Formal Reasoning Approach to Computational Humor
Implementation of premise-based joke generation using abductive reasoning principles.
```

## License

This implementation follows the same license as the parent joker-llm project.

---

**The Abductive Joke Pipeline transforms joke generation from a creative writing problem into a formal reasoning problem, providing a novel and scientifically rigorous approach to computational humor.** 