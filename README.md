# JokePlanSearch: Comprehensive LLM-Based Joke Generation & Evaluation

A sophisticated system for computational humor generation using the PlanSearch methodology, featuring multi-dimensional evaluation with bias mitigation techniques.

## 🎭 Features

### Core Capabilities
- **Multi-Stage Joke Generation**: Topic analysis → angle exploration → structured joke creation → iterative refinement
- **Bias-Minimized Evaluation**: LLM-as-a-judge with position shuffling, ensemble perspectives, and comparative analysis
- **Comprehensive Analysis**: Statistical scoring, confidence intervals, and detailed performance insights
- **Batch Processing**: Handle multiple topics efficiently with parallel processing

### Technical Highlights
- **PlanSearch Methodology**: Break down creative process into discrete, manageable steps
- **Diverse Angle Generation**: Basic humor categories, advanced techniques, and hybrid approaches
- **Multi-Dimensional Scoring**: Cleverness, surprise, relatability, timing, and overall funniness
- **Research-Backed Bias Mitigation**: Random ordering, multiple evaluation rounds, ensemble judging

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd joker-llm

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from joke_plan_search_complete import JokePlanSearchComplete, create_openai_client

# Setup with your API key
api_client = create_openai_client("your-api-key-here")
joke_search = JokePlanSearchComplete(api_client)

# Generate and evaluate jokes
results = joke_search.run_complete_pipeline("artificial intelligence")

# View results
print(results["quick_summary"])
```

### Google Colab Setup

```python
# Install in Colab
!pip install openai anthropic groq pandas tqdm

# Import and setup
from joke_plan_search_complete import JokePlanSearchComplete, setup_for_colab
config = setup_for_colab()

# Quick demo
joke_search = JokePlanSearchComplete()  # Uses mock client for demo
summary = joke_search.run_quick_demo("coffee")
print(summary)
```

## 📖 System Architecture

### Phase 1: Environment Setup
- API key configuration and client initialization
- Bias mitigation parameter configuration
- Rate limiting and error handling setup

### Phase 2: Joke Generation Pipeline
1. **Topic Analysis**: Context understanding, wordplay identification, cultural references
2. **Angle Generation**: 
   - Basic approaches (puns, observational, absurdist, character-based, irony)
   - Advanced techniques (misdirection, meta-humor, callbacks)
   - Hybrid combinations with unexpected elements
3. **Joke Creation**: Structured outline → full joke generation
4. **Refinement**: Critique and improvement through multiple rounds
5. **Diversity Enhancement**: Similarity detection and varied approach generation

### Phase 3: Evaluation Pipeline
1. **Multi-Dimensional Assessment**: 5-point scoring across humor dimensions
2. **Comparative Analysis**: Pairwise comparisons with win/loss tracking
3. **Ensemble Evaluation**: Multiple judge perspectives (casual audience, expert critic)
4. **Bias Detection**: Statistical analysis of scoring patterns and position effects

### Phase 4: Analysis & Reporting
1. **Score Aggregation**: Weighted averaging with confidence intervals
2. **Final Rankings**: Comprehensive joke ranking with statistical measures
3. **Insight Generation**: LLM-powered analysis of successful patterns
4. **Export Options**: JSON, CSV, and formatted text outputs

## 🔧 Configuration Options

### Bias Mitigation Settings
```python
from joke_plan_search_core import BiasConfig

config = BiasConfig()
config.evaluation_rounds = 3
config.judge_temperature = 0.3
config.min_comparisons_per_joke = 5
config.shuffle_iterations = 2
```

### Pipeline Parameters
```python
results = joke_search.run_complete_pipeline(
    "your_topic",
    refinement_rounds=2,           # Number of joke improvement iterations
    top_n=5,                      # Top jokes to analyze in detail
    similarity_threshold=0.7,     # Threshold for detecting similar jokes
    include_bias_analysis=True    # Include statistical bias detection
)
```

## 📊 Output Formats

### Quick Summary
```
JokePlanSearch Results for Topic: 'artificial intelligence'
============================================================
Total jokes generated: 12
Evaluation completed with 5 different scoring dimensions

Top 5 Jokes:
----------------------------------------
1. Why did the AI go to therapy? Because it had too many deep learning issues!
   Score: 8.7 (95% CI: 8.2-9.1)
   Angle: Pun combining AI terminology with mental health
```

### Detailed Report (JSON)
```json
{
  "experiment_summary": {
    "topic": "artificial intelligence",
    "total_jokes_generated": 12,
    "top_score": 8.7,
    "average_score": 6.4
  },
  "ranked_jokes": [...],
  "analysis_insights": "...",
  "performance_metrics": {...},
  "bias_analysis": {...}
}
```

## 🎯 Use Cases

### Research Applications
- **Computational Humor Studies**: Analyze what makes jokes effective
- **Bias Detection Research**: Study LLM evaluation consistency and biases
- **Creative AI Development**: Benchmark joke generation approaches

### Practical Applications
- **Content Creation**: Generate jokes for specific topics or audiences
- **Comedy Writing Assistance**: Explore different humor angles and approaches
- **Educational Tools**: Teach humor theory through systematic analysis

### Experimental Design
- **A/B Testing**: Compare different generation or evaluation strategies
- **Batch Analysis**: Process multiple topics for comparative studies
- **Longitudinal Studies**: Track joke quality improvements over time

## 🔬 Methodology Details

### PlanSearch Implementation
Based on recent research in computational creativity, our implementation breaks joke generation into discrete planning and execution phases:

1. **Strategic Planning**: Analyze topic for humor potential and generate diverse approaches
2. **Tactical Execution**: Convert each approach into structured jokes
3. **Quality Assurance**: Systematic refinement and improvement
4. **Comprehensive Evaluation**: Multi-perspective assessment with bias controls

### Bias Mitigation Techniques
- **Position Randomization**: Shuffle joke order across evaluation rounds
- **Temperature Control**: Lower temperature for evaluation consistency
- **Ensemble Perspectives**: Multiple judge personas (casual, expert)
- **Comparative Validation**: Pairwise comparisons to validate absolute scores
- **Statistical Analysis**: Confidence intervals and bias detection metrics

## 📈 Performance Metrics

The system tracks comprehensive performance data:

- **Generation Efficiency**: API calls, token usage, processing time
- **Evaluation Quality**: Inter-rater reliability, confidence measures
- **Bias Detection**: Position effects, score distributions, consistency analysis
- **Success Rates**: Joke generation success, evaluation completion rates

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

- **New Humor Categories**: Additional joke generation approaches
- **Evaluation Dimensions**: Additional scoring criteria and perspectives
- **Bias Mitigation**: Enhanced techniques for fair evaluation
- **API Integrations**: Support for additional LLM providers
- **Analysis Tools**: New visualization and insight generation methods

## 📚 References

This implementation draws from research in:
- Computational creativity and humor generation
- LLM-as-a-judge methodologies and bias mitigation
- PlanSearch techniques for creative problem solving
- Statistical evaluation and confidence interval estimation

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- [Google Colab Demo](link-to-colab-notebook)
- [API Documentation](link-to-api-docs)
- [Research Paper](link-to-paper)
- [Example Results](link-to-examples)

---

**Ready to generate some laughs with AI? Get started with the quick demo above!** 🎭✨ 