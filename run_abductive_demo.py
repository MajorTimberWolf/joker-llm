#!/usr/bin/env python3
"""
Abductive Joke Pipeline Demo
Demonstrates the complete functionality of the abductive reasoning joke system.
"""

import os
import sys
import logging
from groq_config import GroqConfig, get_recommended_config
from abductive_joke_pipeline import (
    AbductiveJokePipeline, 
    JokeAnalyzer, 
    AbductiveExperimentFramework,
    AbductivePlanSearchIntegration,
    run_abductive_demo
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_groq_client():
    """Setup and return a Groq client with optimized configuration"""
    try:
        import groq
    except ImportError:
        logger.error("Groq SDK not installed. Install with: pip install groq")
        sys.exit(1)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        sys.exit(1)
    
    return groq.Groq(api_key=api_key)


def demo_basic_functionality(client, topic: str = "coffee shops"):
    """Demonstrate basic abductive joke generation"""
    print("\n🎭 === BASIC ABDUCTIVE JOKE GENERATION ===")
    print(f"Topic: {topic}")
    print("-" * 50)
    
    results = run_abductive_demo(topic, client, num_jokes=3)
    
    print(f"✅ Generated {results['jokes_generated']} jokes")
    print(f"📊 Average Logical Consistency: {results['average_consistency']:.2f}/10")
    print(f"🔧 API Calls Used: {results['api_calls_used']}")
    
    for i, joke_data in enumerate(results['jokes']):
        print(f"\n--- Joke {i+1} ---")
        print(f"🏗️  Setup: {joke_data['setup']}")
        print(f"💥 Punchline: {joke_data['punchline']}")
        print(f"🌍 Grounding: {joke_data['joke_world']['grounding_premise']['content']}")
        print(f"🎪 Absurd: {joke_data['joke_world']['absurd_premise']['content']}")
        if results['consistency_scores']:
            print(f"📈 Consistency Score: {results['consistency_scores'][i]:.1f}/10")
    
    return results


def demo_premise_analysis(client, topics: list = ["libraries", "gyms", "social media"]):
    """Demonstrate premise type analysis across multiple topics"""
    print("\n📊 === PREMISE TYPE ANALYSIS ===")
    print(f"Topics: {', '.join(topics)}")
    print("-" * 50)
    
    pipeline = AbductiveJokePipeline(client)
    analyzer = JokeAnalyzer(client)
    
    all_jokes = []
    for topic in topics:
        jokes = pipeline.generate_joke_batch(topic, num_jokes=2)
        all_jokes.extend(jokes)
        print(f"✅ Generated {len(jokes)} jokes for '{topic}'")
    
    # Analyze premise patterns
    analysis = analyzer.analyze_premise_types(all_jokes)
    
    print(f"\n📈 Premise Analysis Results:")
    print(f"Total jokes analyzed: {analysis['total_jokes_analyzed']}")
    
    print("\n🏗️  Grounding Premise Themes:")
    for theme, count in analysis['grounding_themes'].items():
        print(f"  • {theme}: {count}")
    
    print("\n🎪 Absurd Premise Themes:")  
    for theme, count in analysis['absurd_themes'].items():
        print(f"  • {theme}: {count}")
    
    return analysis


def demo_logical_consistency_analysis(client, topic: str = "restaurants"):
    """Demonstrate logical consistency evaluation"""
    print("\n🧠 === LOGICAL CONSISTENCY ANALYSIS ===")
    print(f"Topic: {topic}")
    print("-" * 50)
    
    pipeline = AbductiveJokePipeline(client)
    analyzer = JokeAnalyzer(client)
    
    # Generate jokes
    jokes = pipeline.generate_joke_batch(topic, num_jokes=4)
    
    # Analyze each joke's logical consistency
    consistency_results = []
    
    for i, joke in enumerate(jokes):
        score = analyzer.measure_logical_consistency(joke, joke.joke_world)
        consistency_results.append(score)
        
        print(f"\n--- Joke {i+1} Analysis ---")
        print(f"🏗️  Setup: {joke.setup}")
        print(f"💥 Punchline: {joke.punchline}")
        print(f"🧠 Logical Consistency: {score:.1f}/10")
        
        if score >= 8:
            print("✅ Highly consistent reasoning")
        elif score >= 6:
            print("⚠️  Moderately consistent reasoning")
        else:
            print("❌ Inconsistent reasoning")
    
    avg_consistency = sum(consistency_results) / len(consistency_results)
    print(f"\n📊 Overall Results:")
    print(f"Average Consistency: {avg_consistency:.2f}/10")
    print(f"Best Score: {max(consistency_results):.1f}/10")
    print(f"Worst Score: {min(consistency_results):.1f}/10")
    
    return consistency_results


def demo_experimental_framework(client):
    """Demonstrate the experimental research capabilities"""
    print("\n🔬 === EXPERIMENTAL FRAMEWORK DEMO ===")
    print("-" * 50)
    
    pipeline = AbductiveJokePipeline(client)
    analyzer = JokeAnalyzer(client)
    experiment_framework = AbductiveExperimentFramework(pipeline, analyzer)
    
    # Run premise type experiment
    print("🧪 Running premise type experiment...")
    topics = ["movies", "exercise"]
    premise_results = experiment_framework.run_premise_type_experiment(topics, iterations_per_topic=2)
    
    print(f"✅ Premise experiment completed")
    print(f"Hypothesis: {premise_results['hypothesis']}")
    print(f"Topics tested: {premise_results['topics_tested']}")
    
    for topic, results in premise_results['results_by_topic'].items():
        avg_score = sum(results['abductive_scores']) / len(results['abductive_scores']) if results['abductive_scores'] else 0
        print(f"  • {topic}: {len(results['abductive_jokes'])} jokes, avg score {avg_score:.2f}")
    
    # Run abduction effectiveness experiment
    print("\n🧪 Running abduction effectiveness experiment...")
    effectiveness_results = experiment_framework.run_abduction_effectiveness_experiment(["technology"])
    
    print(f"✅ Effectiveness experiment completed")
    print(f"Hypothesis: {effectiveness_results['hypothesis']}")
    
    for topic, results in effectiveness_results['abductive_results'].items():
        print(f"  • {topic}: {len(results['jokes'])} jokes, avg consistency {results['average_consistency']:.2f}")
    
    return premise_results, effectiveness_results


def demo_export_functionality(client, topic: str = "pets"):
    """Demonstrate export capabilities for human evaluation"""
    print("\n📤 === EXPORT FOR HUMAN EVALUATION ===")
    print(f"Topic: {topic}")
    print("-" * 50)
    
    pipeline = AbductiveJokePipeline(client)
    jokes = pipeline.generate_joke_batch(topic, num_jokes=3)
    
    # Export for human evaluation
    export_data = AbductivePlanSearchIntegration.export_for_evaluation(jokes, format="human_eval")
    
    print(f"✅ Exported {len(export_data['jokes'])} jokes for human evaluation")
    print(f"📋 Evaluation criteria: {', '.join(export_data['evaluation_criteria'])}")
    print(f"📝 Instructions: {export_data['instructions']}")
    
    print("\n📊 Sample Export Data:")
    for joke_data in export_data['jokes'][:2]:  # Show first 2 jokes
        print(f"ID {joke_data['id']}: {joke_data['full_joke']}")
        print(f"  Method: {joke_data['method']}")
        print(f"  Grounding: {joke_data['grounding_premise']}")
        print(f"  Absurd: {joke_data['absurd_premise']}")
        print()
    
    return export_data


def demo_integration_readiness(client):
    """Demonstrate integration with existing PlanSearch pipeline"""
    print("\n🔗 === PLANSEARCH INTEGRATION DEMO ===")
    print("-" * 50)
    
    # Create abductive pipeline
    abductive_pipeline = AbductiveJokePipeline(client)
    
    # Mock existing pipeline for demo
    class MockExistingPipeline:
        def __init__(self):
            self.name = "Traditional PlanSearch Pipeline"
    
    existing_pipeline = MockExistingPipeline()
    
    # Demonstrate integration
    integration_result = AbductivePlanSearchIntegration.integrate_with_plansearch(
        existing_pipeline, abductive_pipeline
    )
    
    print("✅ Integration framework ready")
    print(f"📊 Status: {integration_result['comparison_framework']}")
    print("🔬 Ready for A/B testing between:")
    print(f"  • Traditional: {existing_pipeline.name}")
    print(f"  • Abductive: {type(abductive_pipeline).__name__}")
    
    return integration_result


def run_comprehensive_demo():
    """Run all demonstration features"""
    print("🎭 ABDUCTIVE JOKE PIPELINE - COMPREHENSIVE DEMO")
    print("=" * 60)
    
    # Setup
    client = setup_groq_client()
    config = get_recommended_config("balanced")
    print(f"🔧 Using configuration: {config['model_preference']}")
    print(f"⏱️  Estimated time: {config.get('estimated_time', 'Unknown')}")
    
    try:
        # Basic functionality
        basic_results = demo_basic_functionality(client, "coffee shops")
        
        # Premise analysis
        premise_analysis = demo_premise_analysis(client, ["libraries", "gyms"])
        
        # Logical consistency
        consistency_results = demo_logical_consistency_analysis(client, "restaurants")
        
        # Experimental framework
        premise_exp, effectiveness_exp = demo_experimental_framework(client)
        
        # Export functionality
        export_data = demo_export_functionality(client, "pets")
        
        # Integration readiness
        integration_result = demo_integration_readiness(client)
        
        # Summary
        print("\n🎯 === DEMO SUMMARY ===")
        print("-" * 50)
        print("✅ All systems operational!")
        print(f"📊 Total API calls: {basic_results['api_calls_used']} (basic demo)")
        print(f"🎭 Jokes generated: {basic_results['jokes_generated']} (basic demo)")
        print(f"🧠 Average consistency: {basic_results['average_consistency']:.2f}/10")
        print("\n🔬 Research Capabilities Demonstrated:")
        print("  ✅ Premise generation with grounding + absurd structure")
        print("  ✅ Abductive reasoning for punchline generation")
        print("  ✅ Logical consistency evaluation")
        print("  ✅ Premise type analysis")
        print("  ✅ Experimental framework for hypothesis testing")
        print("  ✅ Export functionality for human evaluation")
        print("  ✅ Integration readiness with existing pipelines")
        
        print("\n🚀 The Abductive Joke Pipeline is ready for research use!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific demo based on argument
        demo_type = sys.argv[1].lower()
        client = setup_groq_client()
        
        if demo_type == "basic":
            demo_basic_functionality(client)
        elif demo_type == "premises":
            demo_premise_analysis(client)
        elif demo_type == "consistency":
            demo_logical_consistency_analysis(client)
        elif demo_type == "experiment":
            demo_experimental_framework(client)
        elif demo_type == "export":
            demo_export_functionality(client)
        elif demo_type == "integration":
            demo_integration_readiness(client)
        else:
            print("Available demos: basic, premises, consistency, experiment, export, integration")
    else:
        # Run comprehensive demo
        run_comprehensive_demo() 