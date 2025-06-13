#!/usr/bin/env python3
"""
Quick test script for local JokePlanSearch setup
"""

import os
from joke_plan_search_complete import (
    JokePlanSearchComplete, 
    create_openai_client,
    create_anthropic_client,
    create_groq_client
)

def test_with_api():
    """Test with actual API."""
    
    # Try different API providers
    api_client = None
    provider = "mock"
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            api_client = create_openai_client(os.getenv("OPENAI_API_KEY"))
            provider = "OpenAI"
            print(f"✅ Connected to {provider}")
        except Exception as e:
            print(f"❌ OpenAI connection failed: {e}")
    
    elif os.getenv("ANTHROPIC_API_KEY"):
        try:
            api_client = create_anthropic_client(os.getenv("ANTHROPIC_API_KEY"))
            provider = "Anthropic"
            print(f"✅ Connected to {provider}")
        except Exception as e:
            print(f"❌ Anthropic connection failed: {e}")
    
    elif os.getenv("GROQ_API_KEY"):
        try:
            api_client = create_groq_client(os.getenv("GROQ_API_KEY"))
            provider = "Groq"
            print(f"✅ Connected to {provider}")
        except Exception as e:
            print(f"❌ Groq connection failed: {e}")
    
    if not api_client:
        print("ℹ️  No API keys found, using mock client for demo")
        api_client = None
        provider = "Mock"
    
    # Initialize system
    joke_search = JokePlanSearchComplete(api_client)
    
    # Run quick test
    print(f"\n🎭 Testing joke generation with {provider}...")
    topic = "Bangalore Traffic"
    
    if provider == "Mock":
        # Use quick demo for mock
        result = joke_search.run_quick_demo(topic)
        print("📝 Demo Result:")
        print(result)
    else:
        # Use full pipeline for real API
        print(f"🔄 Running full pipeline for '{topic}'...")
        results = joke_search.run_complete_pipeline(
            topic,
            refinement_rounds=1,  # Reduced for quick test
            top_n=3,
            include_bias_analysis=False  # Skip for speed
        )
        
        if results["success"]:
            print("✅ Pipeline completed successfully!")
            print(f"⏱️  Execution time: {results['execution_time_seconds']:.2f} seconds")
            print("\n📝 Quick Summary:")
            print(results["quick_summary"][:500] + "..." if len(results["quick_summary"]) > 500 else results["quick_summary"])
        else:
            print(f"❌ Pipeline failed: {results['error']}")

if __name__ == "__main__":
    print("🚀 JokePlanSearch Local Test")
    print("=" * 40)
    test_with_api()
    print("\n✨ Test completed!")
