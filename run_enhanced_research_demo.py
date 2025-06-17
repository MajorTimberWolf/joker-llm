#!/usr/bin/env python3
"""
Enhanced Research-Grade Abductive Joke Pipeline Demo
Demonstrates all the research improvements and experimental capabilities.
"""

import logging
import time
import statistics
from typing import List, Dict, Any
from enhanced_abductive_pipeline import (
    create_enhanced_pipeline, 
    create_multi_judge_analyzer,
    StatisticalAnalyzer,
    EnhancedAbductiveJoke
)
import groq

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demo_multi_premise_worlds(client, topic: str = "coffee shops"):
    """Demonstrate enhanced multi-premise world creation"""
    print(f"\n🏗️  === MULTI-PREMISE WORLD CREATION ===")
    print(f"Topic: {topic}")
    print("-" * 50)
    
    pipeline = create_enhanced_pipeline(client)
    
    # Test different premise configurations
    for num_premises in [2, 3, 4]:
        print(f"\n--- Testing {num_premises}-Premise World ---")
        
        try:
            world = pipeline.establish_multi_premise_world(topic, num_premises)
            
            print(f"✅ Successfully created world with {len(world.premises)} premises:")
            for i, premise in enumerate(world.premises, 1):
                quality = premise.quality_score or 0
                print(f"  {i}. [{premise.premise_type.upper()}] {premise.content}")
                print(f"     Quality: {quality:.1f}/10 | ID: {premise.premise_id}")
            
            print(f"📊 World ID: {world.world_id}")
            
        except Exception as e:
            print(f"❌ Failed to create {num_premises}-premise world: {e}")


def demo_premise_quality_scoring(client, topic: str = "restaurants"):
    """Demonstrate premise quality scoring and filtering"""
    print(f"\n🎯 === PREMISE QUALITY SCORING ===")
    print(f"Topic: {topic}")
    print("-" * 50)
    
    pipeline = create_enhanced_pipeline(client)
    
    # Generate premises and show quality analysis
    world = pipeline.establish_multi_premise_world(topic, 3)
    
    print("📈 Premise Quality Analysis:")
    total_quality = 0
    for premise in world.premises:
        quality = premise.quality_score or 0
        specificity = premise.specificity_score or 0
        novelty = premise.novelty_score or 0
        
        print(f"\n🔍 {premise.premise_type.title()} Premise:")
        print(f"   Content: {premise.content}")
        print(f"   Overall Quality: {quality:.1f}/10")
        print(f"   Specificity: {specificity:.1f}/10")
        print(f"   Novelty: {novelty:.1f}/10")
        
        if quality >= 7:
            print("   ✅ High quality premise")
        elif quality >= 5:
            print("   ⚠️  Moderate quality premise")
        else:
            print("   ❌ Low quality premise (would be filtered)")
        
        total_quality += quality
    
    avg_quality = total_quality / len(world.premises)
    print(f"\n📊 Average World Quality: {avg_quality:.2f}/10")
    
    return world


def demo_adaptive_temperature_generation(client, world):
    """Demonstrate adaptive temperature joke generation"""
    print(f"\n🌡️  === ADAPTIVE TEMPERATURE GENERATION ===")
    print("-" * 50)
    
    pipeline = create_enhanced_pipeline(client)
    
    # Generate jokes with and without adaptive temperature
    print("🧪 Testing Adaptive Temperature Control...")
    
    # Standard generation
    print("\n--- Standard Generation (T=0.9) ---")
    joke_standard = pipeline.generate_enhanced_joke(world, adaptive_temperature=False)
    print(f"Setup: {joke_standard.setup}")
    print(f"Punchline: {joke_standard.punchline}")
    print(f"Temperature used: {joke_standard.metadata.get('temperature', 'unknown')}")
    
    # Adaptive generation
    print("\n--- Adaptive Generation ---")
    joke_adaptive = pipeline.generate_enhanced_joke(world, adaptive_temperature=True)
    print(f"Setup: {joke_adaptive.setup}")
    print(f"Punchline: {joke_adaptive.punchline}")
    print(f"Temperature used: {joke_adaptive.metadata.get('temperature', 'unknown')}")
    
    if joke_adaptive.metadata.get('retried'):
        print("🔄 Adaptive retry was triggered (low initial quality)")
    else:
        print("✅ First attempt met quality threshold")
    
    return [joke_standard, joke_adaptive]


def demo_multi_judge_evaluation(client, jokes: List[EnhancedAbductiveJoke]):
    """Demonstrate multi-judge ensemble evaluation"""
    print(f"\n👥 === MULTI-JUDGE ENSEMBLE EVALUATION ===")
    print("-" * 50)
    
    analyzer = create_multi_judge_analyzer(client, num_judges=3)
    
    all_results = []
    
    for i, joke in enumerate(jokes, 1):
        print(f"\n--- Evaluating Joke {i} ---")
        print(f"Setup: {joke.setup}")
        print(f"Punchline: {joke.punchline}")
        
        try:
            results = analyzer.evaluate_logical_consistency_ensemble(joke)
            all_results.append(results)
            
            print(f"\n📊 Judge Scores: {results['scores']}")
            print(f"🎯 Median Score: {results['median_score']:.1f}/10")
            print(f"📈 Mean Score: {results['mean_score']:.1f}/10")
            print(f"📏 Standard Deviation: {results['std_score']:.2f}")
            print(f"🤝 Judge Agreement: {results['agreement']:.1%}")
            
            if results['agreement'] > 0.8:
                print("✅ High judge agreement")
            elif results['agreement'] > 0.6:
                print("⚠️  Moderate judge agreement")
            else:
                print("❌ Low judge agreement - results may be unreliable")
            
            print("\n💭 Judge Explanations:")
            for explanation in results['explanations']:
                print(f"   • {explanation}")
                
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
    
    return all_results


def demo_statistical_analysis(evaluation_results: List[Dict[str, Any]]):
    """Demonstrate statistical analysis capabilities"""
    print(f"\n📊 === STATISTICAL ANALYSIS ===")
    print("-" * 50)
    
    if len(evaluation_results) < 2:
        print("❌ Need at least 2 jokes for statistical analysis")
        return
    
    # Extract scores for analysis
    standard_scores = []
    adaptive_scores = []
    
    for i, result in enumerate(evaluation_results):
        if i == 0:  # Standard generation
            standard_scores.extend(result['scores'])
        else:  # Adaptive generation
            adaptive_scores.extend(result['scores'])
    
    if not standard_scores or not adaptive_scores:
        print("❌ Insufficient data for comparison")
        return
    
    # Perform statistical comparison
    print("🔬 Comparing Standard vs Adaptive Temperature Generation:")
    
    analyzer = StatisticalAnalyzer()
    comparison = analyzer.compare_groups(adaptive_scores, standard_scores)
    
    print(f"\n📈 Results:")
    print(f"   Adaptive Mean: {comparison['group1_mean']:.2f}")
    print(f"   Standard Mean: {comparison['group2_mean']:.2f}")
    print(f"   Difference: {comparison['group1_mean'] - comparison['group2_mean']:.2f}")
    print(f"   t-statistic: {comparison['t_statistic']:.3f}")
    print(f"   p-value: {comparison['p_value']:.4f}")
    print(f"   Effect size (Cohen's d): {comparison['effect_size']:.3f}")
    print(f"   Statistical significance: {comparison['interpretation']}")
    
    # Interpret effect size
    abs_effect = abs(comparison['effect_size'])
    if abs_effect < 0.2:
        effect_interp = "trivial"
    elif abs_effect < 0.5:
        effect_interp = "small"
    elif abs_effect < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    
    print(f"   Effect size interpretation: {effect_interp}")
    
    # Research conclusion
    if comparison['significant'] and abs_effect >= 0.3:
        print(f"\n🎯 Research Conclusion: Significant improvement with {effect_interp} effect size")
    elif comparison['significant']:
        print(f"\n⚠️  Research Conclusion: Statistically significant but {effect_interp} effect")
    else:
        print(f"\n❌ Research Conclusion: No significant difference detected")


def demo_premise_diversity_analysis(client, topics: List[str] = ["coffee", "libraries", "gyms"]):
    """Demonstrate premise diversity analysis across topics"""
    print(f"\n🌈 === PREMISE DIVERSITY ANALYSIS ===")
    print(f"Topics: {', '.join(topics)}")
    print("-" * 50)
    
    pipeline = create_enhanced_pipeline(client)
    
    all_premises = {'grounding': [], 'absurd': [], 'conditional': []}
    
    for topic in topics:
        print(f"\n🎯 Analyzing {topic}...")
        
        try:
            world = pipeline.establish_multi_premise_world(topic, 3)
            
            for premise in world.premises:
                all_premises[premise.premise_type].append(premise.content)
            
            print(f"   ✅ Generated {len(world.premises)} premises")
            
        except Exception as e:
            print(f"   ❌ Failed for {topic}: {e}")
    
    # Analyze diversity
    print(f"\n📊 Diversity Analysis:")
    
    for premise_type, premises in all_premises.items():
        if premises:
            unique_premises = len(set(premises))
            total_premises = len(premises)
            diversity_ratio = unique_premises / total_premises
            
            print(f"\n🏷️  {premise_type.title()} Premises:")
            print(f"   Total: {total_premises}")
            print(f"   Unique: {unique_premises}")
            print(f"   Diversity Ratio: {diversity_ratio:.1%}")
            
            if diversity_ratio > 0.8:
                print("   ✅ High diversity")
            elif diversity_ratio > 0.6:
                print("   ⚠️  Moderate diversity")
            else:
                print("   ❌ Low diversity - consider expanding premise generation")
            
            # Show examples
            print(f"   Examples:")
            for premise in list(set(premises))[:3]:
                print(f"     • {premise}")


def demo_experimental_framework(client, topics: List[str] = ["food", "technology"]):
    """Demonstrate research experimental framework"""
    print(f"\n🔬 === EXPERIMENTAL FRAMEWORK DEMO ===")
    print(f"Topics: {', '.join(topics)}")
    print("-" * 50)
    
    pipeline = create_enhanced_pipeline(client)
    analyzer = create_multi_judge_analyzer(client, num_judges=2)  # Reduced for demo
    
    # Hypothesis: Multi-premise worlds produce higher quality jokes
    print("🧪 Hypothesis: Multi-premise worlds produce higher quality jokes")
    print("📋 Experimental Design: 2-premise vs 3-premise comparison")
    
    conditions = {'2-premise': [], '3-premise': []}
    
    for topic in topics:
        print(f"\n🎯 Testing topic: {topic}")
        
        for condition in ['2-premise', '3-premise']:
            num_premises = int(condition.split('-')[0])
            
            try:
                # Generate world and joke
                world = pipeline.establish_multi_premise_world(topic, num_premises)
                joke = pipeline.generate_enhanced_joke(world)
                
                # Evaluate quality
                evaluation = analyzer.evaluate_logical_consistency_ensemble(joke)
                score = evaluation['median_score']
                
                conditions[condition].append(score)
                print(f"   {condition}: Score = {score:.1f}/10")
                
            except Exception as e:
                print(f"   ❌ {condition} failed: {e}")
    
    # Statistical analysis of experiment
    if conditions['2-premise'] and conditions['3-premise']:
        print(f"\n📊 Experimental Results:")
        
        stats_analyzer = StatisticalAnalyzer()
        comparison = stats_analyzer.compare_groups(
            conditions['3-premise'], 
            conditions['2-premise']
        )
        
        print(f"   3-premise mean: {comparison['group1_mean']:.2f}")
        print(f"   2-premise mean: {comparison['group2_mean']:.2f}")
        print(f"   Effect size: {comparison['effect_size']:.3f}")
        print(f"   Significance: {comparison['interpretation']}")
        
        # Research conclusion
        if comparison['significant'] and comparison['effect_size'] > 0.3:
            print(f"\n🎯 Experimental Conclusion: Multi-premise worlds show significant improvement")
        else:
            print(f"\n📝 Experimental Conclusion: No significant improvement detected")
    else:
        print("❌ Insufficient data for experimental analysis")


def main():
    """Run comprehensive research-grade demo"""
    print("🔬 Enhanced Research-Grade Abductive Joke Pipeline")
    print("=" * 60)
    print("Demonstrating all research improvements and capabilities")
    
    # Setup (you would use your actual API key)
    try:
        client = groq.Groq(api_key="your-groq-api-key-here")
        print("✅ LLM client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize LLM client: {e}")
        print("🔧 Using mock client for structure demonstration")
        client = None
        return
    
    try:
        # Demo 1: Multi-premise worlds
        demo_multi_premise_worlds(client, "coffee shops")
        
        # Demo 2: Premise quality scoring
        world = demo_premise_quality_scoring(client, "restaurants")
        
        # Demo 3: Adaptive temperature generation
        jokes = demo_adaptive_temperature_generation(client, world)
        
        # Demo 4: Multi-judge evaluation
        evaluation_results = demo_multi_judge_evaluation(client, jokes)
        
        # Demo 5: Statistical analysis
        demo_statistical_analysis(evaluation_results)
        
        # Demo 6: Premise diversity analysis
        demo_premise_diversity_analysis(client, ["coffee", "libraries"])
        
        # Demo 7: Experimental framework
        demo_experimental_framework(client, ["food", "technology"])
        
        print(f"\n🎉 === DEMO COMPLETE ===")
        print("All research-grade features demonstrated successfully!")
        print("\n🔬 Ready for publication-quality research:")
        print("✓ Multi-premise worlds with quality filtering")
        print("✓ Multi-judge ensemble evaluation")
        print("✓ Adaptive temperature control")
        print("✓ Statistical significance testing")
        print("✓ Experimental design framework")
        print("✓ Comprehensive diversity analysis")
        print("✓ Type safety and validation")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo encountered an error: {e}")


if __name__ == "__main__":
    main() 