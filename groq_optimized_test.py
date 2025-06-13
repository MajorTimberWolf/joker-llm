#!/usr/bin/env python3
"""
Groq-Optimized JokePlanSearch Test Script
Optimized for Groq's 30 requests/minute rate limit
"""

import os
import time
from dotenv import load_dotenv #type: ignore
from joke_plan_search_complete import JokePlanSearchComplete, create_groq_client, BiasConfig

# Load environment variables
load_dotenv()

def create_groq_optimized_config():
    """Create configuration optimized for Groq rate limits."""
    config = BiasConfig()
    
    # Reduce API calls while maintaining quality
    config.evaluation_rounds = 2  # Reduced from 3
    config.min_comparisons_per_joke = 2  # Reduced from 3
    
    return config

def run_groq_optimized_pipeline():
    """Run a Groq-optimized joke generation pipeline."""
    
    print("🚀 Groq-Optimized JokePlanSearch")
    print("=" * 50)
    print("⚡ Optimized for 30 requests/minute rate limit")
    print("🎯 Reduced API calls while maintaining quality")
    
    # Check for Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY environment variable not found!")
        print("Please set your Groq API key:")
        print("export GROQ_API_KEY='your-groq-api-key-here'")
        return
    
    try:
        # Create Groq client
        print("\n🔗 Connecting to Groq API...")
        api_client = create_groq_client(groq_api_key)
        
        # Create optimized configuration
        groq_config = create_groq_optimized_config()
        
        # Initialize JokePlanSearch with optimized config
        joke_search = JokePlanSearchComplete(api_client, groq_config)
        print("✅ Connected to Groq with optimized rate limiting!")
        
        # Test topic
        topic = input("\nEnter a topic for joke generation (or press Enter for 'AI programming'): ").strip()
        if not topic:
            topic = "AI programming"
        
        print(f"\n🎭 Generating jokes about '{topic}' using Groq...")
        print("⏱️  Estimated time: ~3-4 minutes (due to rate limiting)")
        
        start_time = time.time()
        
        # Run optimized pipeline
        results = joke_search.run_complete_pipeline(
            topic,
            refinement_rounds=1,        # Reduced from 2 to save API calls
            top_n=3,                   # Reduced from 5 to save API calls
            include_bias_analysis=False # Skip to save API calls
        )
        
        end_time = time.time()
        
        if results["success"]:
            print(f"\n✅ Groq pipeline completed successfully!")
            print(f"⏱️  Execution time: {end_time - start_time:.1f} seconds")
            print(f"🔢 API calls made: {joke_search.api_call_count}")
            print(f"🎯 Tokens used: {joke_search.total_tokens_used}")
            
            # Calculate rate info
            calls_per_minute = (joke_search.api_call_count / (end_time - start_time)) * 60
            print(f"📊 Rate: {calls_per_minute:.1f} calls/minute (limit: 30)")
            
            print("\n" + "="*60)
            print("📝 GROQ-OPTIMIZED RESULTS")
            print("="*60)
            print(results["quick_summary"])
            
            # Save results
            filename = f"{topic.replace(' ', '_')}_groq_optimized.json"
            joke_search.save_results_to_json(filename)
            print(f"\n💾 Results saved to {filename}")
            
        else:
            print(f"❌ Pipeline failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your API key is valid")
        print("2. Ensure you have Groq credits available")
        print("3. Try again in a few minutes if rate limited")

def run_quick_groq_test():
    """Run a minimal test to check Groq connection."""
    
    print("\n🧪 Quick Groq Connection Test")
    print("-" * 30)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ No Groq API key found")
        return False
    
    try:
        import groq
        client = groq.Groq(api_key=groq_api_key)
        
        # Test with faster model for initial test
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Faster model for testing
            messages=[
                {"role": "user", "content": "Generate one short programming joke"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        print("✅ Groq API connection successful!")
        print(f"Test joke: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

def show_groq_optimization_tips():
    """Show tips for optimizing Groq usage."""
    print("\n💡 Groq Optimization Tips:")
    print("=" * 40)
    print("🏃 Fast Models (Higher Rate Limits):")
    print("  • llama-3.1-8b-instant (6,000 tokens/min)")
    print("  • gemma2-9b-it (15,000 tokens/min)")
    print("  • meta-llama/llama-4-scout-17b-16e-instruct (30,000 tokens/min)")
    print("\n🎯 Quality Models (Lower Rate Limits):")
    print("  • llama-3.3-70b-versatile (12,000 tokens/min)")
    print("  • deepseek-r1-distill-llama-70b (6,000 tokens/min)")
    print("\n⚙️  Rate Limit Strategies:")
    print("  • Use 2+ second delays between calls")
    print("  • Reduce refinement rounds")
    print("  • Use faster models for initial generation")
    print("  • Batch similar operations")
    print("  • Consider parallel processing with multiple API keys")

if __name__ == "__main__":
    # Test connection first
    if run_quick_groq_test():
        # If connection works, offer full pipeline
        choice = input("\nRun full optimized pipeline? (y/n): ").strip().lower()
        if choice == 'y':
            run_groq_optimized_pipeline()
    
    show_groq_optimization_tips()
    print("\n✨ Test completed!") 