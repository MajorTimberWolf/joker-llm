#!/usr/bin/env python3
"""
JokePlanSearch Pipeline Runner
Handles different API providers and configurations with real-time optimization.
"""

import os
from joke_plan_search_complete import JokePlanSearchComplete
from groq_config import get_recommended_config, GroqModelSelector

def check_api_keys():
    """Check which API keys are available."""
    groq_key = os.getenv('GROQ_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY') 
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    return {
        'groq': groq_key,
        'openai': openai_key,
        'anthropic': anthropic_key
    }

def run_optimized_pipeline(topic: str, optimization_level: str = "free"):
    """
    Run JokePlanSearch with different optimization levels.
    
    Args:
        topic: Topic for joke generation
        optimization_level: "free", "balanced", "quality"
    """
    print(f"🎯 Running {optimization_level} optimization for topic: '{topic}'")
    
    # Get optimized configuration
    config = get_recommended_config(optimization_level)
    print(f"📊 Using {config['model_preference']} model")
    print(f"⏱️  Estimated: {config['estimated_calls']} API calls in {config['estimated_time']}")
    
    # Initialize with optimized settings
    joke_search = JokePlanSearchComplete(
        bias_config=config["bias_config"]
    )
    
    # Apply pipeline settings
    pipeline_settings = config.get("pipeline_settings", {})
    
    # Run the pipeline with optimized parameters
    result = joke_search.run_complete_pipeline(
        topic=topic,
        top_n=pipeline_settings.get("top_n", 5),
        refinement_rounds=pipeline_settings.get("refinement_rounds", 0),
        include_bias_analysis=pipeline_settings.get("include_bias_analysis", False),
        num_angles=pipeline_settings.get("num_angles", 5),
        max_jokes=pipeline_settings.get("max_jokes", 5)
    )
    
    return result

if __name__ == "__main__":
    # Check available API keys
    api_keys = check_api_keys()
    
    if api_keys['groq']:
        print("🔗 Groq API key detected")
        
        # Get user input for topic
        topic = input("Enter a topic for joke generation: ").strip()
        if not topic:
            topic = "penguins"  # Default topic
            
        # Ask user for optimization level
        print("\n🎛️  Choose optimization level:")
        print("1. FREE TIER (12 calls, 45-60s) - Recommended for Groq free tier")
        print("2. BALANCED (25 calls, 2-3 min) - Good quality/speed balance")  
        print("3. QUALITY (40 calls, 4-5 min) - Best results, more API usage")
        
        choice = input("Enter choice (1/2/3) or press Enter for FREE TIER: ").strip()
        
        optimization_levels = {
            "1": "free",
            "2": "balanced", 
            "3": "quality",
            "": "free"  # Default
        }
        
        optimization = optimization_levels.get(choice, "free")
        
        try:
            result = run_optimized_pipeline(topic, optimization)
            print("\n" + "="*50)
            print(result)
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            print("\n💡 Try using FREE TIER mode (option 1) to reduce API usage")
            
    elif api_keys['openai']:
        print("🔗 Using OpenAI API")
        topic = input("Enter a topic for joke generation: ").strip()
        if not topic:
            topic = "penguins"
            
        joke_search = JokePlanSearchComplete()
        result = joke_search.run_complete_pipeline(topic)
        print("\n" + "="*50)
        print(result)
        
    elif api_keys['anthropic']:
        print("🔗 Using Anthropic API")  
        topic = input("Enter a topic for joke generation: ").strip()
        if not topic:
            topic = "penguins"
            
        joke_search = JokePlanSearchComplete()
        result = joke_search.run_complete_pipeline(topic)
        print("\n" + "="*50)
        print(result)
        
    else:
        print("❌ No valid API key detected. Please set one of the following environment variables before running this script:")
        print("   • GROQ_API_KEY  (recommended)\\n   • OPENAI_API_KEY\\n   • ANTHROPIC_API_KEY")
        print("\\n🆓 For free tier users, Groq offers the best rate limits!")
        print("   Get your free API key at: https://console.groq.com/keys")
        exit(1)
