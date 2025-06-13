# JokePlanSearch: LLM-Based Joke Generation with PlanSearch

A sophisticated system for computational humor generation using the PlanSearch methodology, featuring multi-dimensional evaluation with bias mitigation techniques. This project implements a structured approach to creative text generation, breaking down the process into planning, generation, and evaluation stages.

## 🎭 Features

### Core Capabilities
- **PlanSearch Methodology**: Implements a PlanSearch algorithm to first generate diverse "primitive observations" about a topic, then combines them into "combinatorial plans" to guide joke creation.
- **Structured Joke Generation**: Jokes are generated based on explicit plans, allowing for more controlled and varied outputs.
- **LLM-as-a-Judge Evaluation**: A robust evaluation pipeline that uses an LLM to score jokes based on multiple criteria.
- **Bias-Minimized Evaluation**: Incorporates techniques like position shuffling to mitigate positional bias in LLM-based evaluations.
- **Comprehensive Analysis**: Provides detailed analysis of joke quality and performance.
- **Modular Pipeline**: The entire process is broken down into a clear, modular pipeline from planning to final analysis.

### Technical Highlights
- **Discrete Creative Steps**: Separates the creative process into Planning → Generation → Evaluation.
- **Combinatorial Creativity**: Explores a wide creative space by combining smaller ideas (observations) into larger joke concepts (plans).
- **Multi-Dimensional Scoring**: Evaluates jokes based on humor, relevance, and originality.
- **Research-Backed Bias Mitigation**: Uses random ordering and multiple evaluation rounds to ensure fair assessment.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/joker-llm.git
cd joker-llm

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

The main entry point for running the system is `run_joke_search.py`. This script executes the entire pipeline.

```bash
# Run the complete joke generation and evaluation pipeline
python run_joke_search.py --topic "artificial intelligence"
```

You can also import and run the pipeline from another Python script:

```python
from joke_plansearch_pipeline import run_joke_generation_pipeline

# Define the topic
topic = "coffee"

# Run the pipeline
final_report = run_joke_generation_pipeline(topic)

# View the results
print(final_report)
```

## 📖 System Architecture

The system is designed as a multi-stage pipeline that implements the PlanSearch methodology.

### Phase 1: Primitive Observation Generation
- The process starts by generating a set of "primitive observations" about the given topic. These are small, distinct ideas that serve as building blocks for jokes.
- **[Read more about Phase 1](./phase1_observations.md)**

### Phase 2: Combinatorial Plan Generation
- The primitive observations are then combined to form "combinatorial plans," which are high-level concepts for jokes.
- **[Read more about Phase 2](./phase2_plans.md)**

### Phase 3: Joke Generation
- For each plan, the system uses an LLM to generate a joke, translating the concept into a final text.
- **[Read more about Phase 3](./phase3_generation.md)**

### Phase 4: Joke Evaluation
- The generated jokes are evaluated by an LLM-as-a-Judge, which scores them on multiple criteria while mitigating for positional bias.
- **[Read more about Phase 4](./phase4_evaluation.md)**

### Phase 5: Analysis & Reporting
- The evaluation scores are aggregated, and a final, ranked report is generated in JSON format.
- **[Read more about Phase 5](./phase5_analysis.md)**

## 🔧 Core Components

-   `joke_plansearch_pipeline.py`: Orchestrates the entire pipeline from start to finish.
-   `joke_plan_search.py`: Contains the core `JokePlanSearch` class that implements the PlanSearch logic (observations, plans, joke generation).
-   `joke_evaluation.py`: Handles the LLM-as-a-Judge evaluation process.
-   `joke_analysis.py`: Provides tools for analyzing the results.
-   `run_joke_search.py`: A command-line script to easily run the pipeline.

## 📊 Output Format

The final output is a JSON report containing a summary, the best joke, and a ranked list of all generated jokes with their scores and the plans that created them.

```json
{
  "best_joke": {
    "joke": "Why did the coffee file a police report? It got mugged!",
    "score": 9,
    "plan": "A pun combining 'mug' as a cup and 'mugged' as being robbed."
  },
  "ranked_jokes": [
    {
      "joke": "Why did the coffee file a police report? It got mugged!",
      "score": 9,
      "plan": "A pun combining 'mug' as a cup and 'mugged' as being robbed."
    }
  ],
  "summary": {
    "topic": "coffee",
    "total_jokes_generated": 10,
    "average_score": 7.5
  }
}
```

## 🎯 Use Cases

### Research Applications
- **Computational Creativity**: Studying structured approaches to creative generation.
- **Prompt Engineering**: Analyzing the impact of structured vs. unstructured prompts.
- **LLM Evaluation**: Researching and mitigating biases in LLM-based assessment.

### Practical Applications
- **Content Creation**: A tool for generating creative and diverse jokes for any topic.
- **Comedy Writing Assistant**: Helping writers explore a wide range of humor angles systematically.

## 📄 License

MIT License - see the `LICENSE` file for details.

## 🔗 Resources [WIP]


- [Research Paper](link-to-paper)
- [Example Results](link-to-examples)