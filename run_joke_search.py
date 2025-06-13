#!/usr/bin/env python3
"""
Simple JokePlanSearch usage script
"""

import os
from joke_plan_search_complete import JokePlanSearchComplete, create_openai_client, create_groq_client, create_anthropic_client

def main():
    # Get topic from user
    topic = input("Enter a topic for joke generation: ").strip()
    if not topic:
        topic = "programming"  # Default topic
    
    # Try different API providers in order of preference
    api_client = None
    provider = None
    
    # Check for Groq first (fastest and cheapest)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            api_client = create_groq_client(groq_key)
            provider = "Groq"
            print(f"🔗 Using Groq API for '{topic}' (Llama 3.1 70B)")
        except Exception as e:
            print(f"❌ Groq connection failed: {e}")
    
    # Fallback to OpenAI
    if not api_client:
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                api_client = create_openai_client(openai_key)
                provider = "OpenAI"
                print(f"🔗 Using OpenAI API for '{topic}' (GPT-4)")
            except Exception as e:
                print(f"❌ OpenAI connection failed: {e}")
    
    # Fallback to Anthropic
    if not api_client:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                api_client = create_anthropic_client(anthropic_key)
                provider = "Anthropic"
                print(f"🔗 Using Anthropic API for '{topic}' (Claude 3)")
            except Exception as e:
                print(f"❌ Anthropic connection failed: {e}")
    
    if api_client:
        # Use real API
        joke_search = JokePlanSearchComplete(api_client)
        print(f"⚡ Running full pipeline with {provider}...")
        
        # Run full pipeline
        results = joke_search.run_complete_pipeline(topic)
        
        if results["success"]:
            print("\n" + "="*50)
            print(results["quick_summary"])
            
            # Save results with provider name
            filename = f"{topic.replace(' ', '_')}_{provider.lower()}_results.json"
            joke_search.save_results_to_json(filename)
            print(f"\n💾 Detailed results saved to {filename}")
            
            # Show performance stats
            print(f"\n📊 Performance:")
            print(f"   • Provider: {provider}")
            print(f"   • Execution time: {results['execution_time_seconds']:.2f}s")
            print(f"   • API calls: {joke_search.api_call_count}")
            print(f"   • Tokens used: {joke_search.total_tokens_used}")
        else:
            print(f"❌ Error: {results['error']}")
    else:
        # Use mock client
        joke_search = JokePlanSearchComplete()
        print(f"🎭 No API keys found, using mock client for demo with '{topic}'")
        result = joke_search.run_quick_demo(topic)
        print("\n" + "="*50)
        print(result)
        print("\n💡 To use real APIs, set one of these environment variables:")
        print("   export GROQ_API_KEY='your-groq-key'")
        print("   export OPENAI_API_KEY='your-openai-key'")
        print("   export ANTHROPIC_API_KEY='your-anthropic-key'")

if __name__ == "__main__":
    main()
