#!/usr/bin/env python3
"""
Groq API Test Script for JokePlanSearch
Test script specifically for Groq API integration
"""

import os
import sys
from dotenv import load_dotenv
from joke_plan_search_complete import JokePlanSearchComplete, create_groq_client

# Load environment variables
load_dotenv()

def test_groq_api():
    """Test specifically with Groq API."""
    
    print("🚀 JokePlanSearch Groq API Test")
    print("=" * 50)
    
    # Check for Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY environment variable not found!")
        print("Please set your Groq API key:")
        print("export GROQ_API_KEY='your-groq-api-key-here'")
        return
    
    try:
        # Create Groq client
        print("🔗 Connecting to Groq API...")
        api_client = create_groq_client(groq_api_key)
        print("✅ Connected to Groq successfully!")
        
        # Initialize JokePlanSearch with Groq client
        joke_search = JokePlanSearchComplete(api_client)
        
        # Test topic
        topic = input("\nEnter a topic for joke generation (or press Enter for 'programming'): ").strip()
        if not topic:
            topic = "programming"
        
        print(f"\n🎭 Generating jokes about '{topic}' using Groq...")
        print("⚡ Using Groq's fast Llama model for generation...")
        
        # Run the complete pipeline with Groq
        results = joke_search.run_complete_pipeline(
            topic,
            refinement_rounds=2,  # Keep moderate for speed
            top_n=5,             # Generate 5 jokes
            include_bias_analysis=True
        )
        
        if results["success"]:
            print("\n✅ Groq pipeline completed successfully!")
            print(f"⏱️  Execution time: {results['execution_time_seconds']:.2f} seconds")
            print(f"🔢 API calls made: {joke_search.api_call_count}")
            print(f"🎯 Tokens used: {joke_search.total_tokens_used}")
            
            print("\n" + "="*60)
            print("📝 GROQ GENERATION RESULTS")
            print("="*60)
            print(results["quick_summary"])
            
            # Save results
            filename = f"{topic.replace(' ', '_')}_groq_results.json"
            joke_search.save_results_to_json(filename)
            print(f"\n💾 Detailed results saved to {filename}")
            
            # Show performance metrics
            metrics = joke_search.get_performance_metrics()
            print(f"\n📊 Performance Metrics:")
            print(f"   • Total API calls: {metrics['total_api_calls']}")
            print(f"   • Total tokens: {metrics['total_tokens']}")
            print(f"   • Avg response time: {metrics.get('avg_response_time', 'N/A')}")
            
        else:
            print(f"❌ Pipeline failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Error during Groq API test: {str(e)}")
        print("Common issues:")
        print("1. Invalid API key")
        print("2. Network connectivity")
        print("3. Rate limiting")
        print("4. Missing groq library (run: pip install groq)")

def test_simple_groq_call():
    """Test a simple Groq API call first."""
    
    print("\n🧪 Testing simple Groq API call...")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ No Groq API key found")
        return
    
    try:
        import groq
        client = groq.Groq(api_key=groq_api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": "Tell me a short joke about programming"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        print("✅ Simple Groq API call successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Simple Groq API call failed: {str(e)}")
        return False

def show_groq_models():
    """Show available Groq models."""
    print("\n📋 Available Groq Models:")
    print("• llama-3.3-70b-versatile (Default - High Quality)")
    print("• llama-3.1-8b-instant (Fastest)")
    print("• gemma2-9b-it (Google's Gemma)")
    print("• qwen-qwq-32b (Reasoning model)")
    print("• deepseek-r1-distill-llama-70b (Reasoning)")
    print("• mistral-saba-24b (Multilingual)")

if __name__ == "__main__":
    # Test simple call first
    if test_simple_groq_call():
        # If simple call works, run full test
        test_groq_api()
    
    show_groq_models()
    print("\n✨ Groq test completed!") 