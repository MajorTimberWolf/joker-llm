"""
JokePlanSearch: Example Usage Script
Demonstrates how to use the complete joke generation and evaluation system.
"""

import os
import json
from joke_plan_search_complete import (
    JokePlanSearchComplete, 
    create_openai_client,
    create_anthropic_client,
    setup_for_colab
)

def example_basic_usage():
    """Basic usage example with OpenAI."""
    print("=== Basic Usage Example ===")
    
    # Setup API client (replace with your actual API key)
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    if api_key == "your-api-key-here":
        print("Please set your OPENAI_API_KEY environment variable")
        print("Using mock client for demonstration...")
        # Use mock client for demo
        joke_search = JokePlanSearchComplete()
    else:
        api_client = create_openai_client(api_key)
        joke_search = JokePlanSearchComplete(api_client)
    
    # Run the complete pipeline
    topic = "artificial intelligence"
    print(f"\nGenerating jokes for topic: '{topic}'")
    
    results = joke_search.run_complete_pipeline(topic)
    
    if results["success"]:
        print("\n" + results["quick_summary"])
        print(f"\nPipeline completed in {results['execution_time_seconds']:.2f} seconds")
    else:
        print(f"Pipeline failed: {results['error']}")

def example_quick_demo():
    """Quick demo example."""
    print("\n=== Quick Demo Example ===")
    
    # Use mock client for quick demo
    joke_search = JokePlanSearchComplete()
    
    topics = ["coffee", "programming", "cats"]
    
    for topic in topics:
        print(f"\nQuick demo for '{topic}':")
        summary = joke_search.run_quick_demo(topic)
        print(summary[:200] + "..." if len(summary) > 200 else summary)

def example_batch_processing():
    """Batch processing example."""
    print("\n=== Batch Processing Example ===")
    
    joke_search = JokePlanSearchComplete()
    
    topics = ["space exploration", "cooking", "social media"]
    
    print(f"Processing {len(topics)} topics in batch mode...")
    
    # Run batch processing with reduced parameters for speed
    batch_results = joke_search.batch_process_topics(
        topics,
        refinement_rounds=1,
        top_n=3,
        include_bias_analysis=False
    )
    
    # Export results in different formats
    print("\nBatch results (CSV format):")
    csv_output = joke_search.export_batch_results(batch_results, "csv")
    print(csv_output)

def example_custom_configuration():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    from joke_plan_search_core import BiasConfig
    
    # Create custom bias configuration
    custom_config = BiasConfig()
    custom_config.evaluation_rounds = 2
    custom_config.min_comparisons_per_joke = 2
    custom_config.judge_temperature = 0.2
    
    joke_search = JokePlanSearchComplete(bias_config=custom_config)
    
    # Run with custom parameters
    results = joke_search.run_complete_pipeline(
        "virtual reality",
        refinement_rounds=1,
        top_n=3,
        similarity_threshold=0.8,
        include_bias_analysis=True
    )
    
    if results["success"]:
        # Show performance metrics
        metrics = results["pipeline_results"]["performance_metrics"]
        print(f"API calls made: {metrics['api_calls_made']}")
        print(f"Jokes generated: {metrics['jokes_generated']}")
        print(f"Successful evaluations: {metrics['successful_evaluations']}")

def example_colab_setup():
    """Example for Google Colab environment."""
    print("\n=== Google Colab Setup Example ===")
    
    # Setup for Colab
    config = setup_for_colab()
    print("Colab configuration loaded")
    
    # Create system with Colab-optimized settings
    joke_search = JokePlanSearchComplete(bias_config=config["bias_config"])
    
    # Run with default Colab parameters
    results = joke_search.run_complete_pipeline(
        "machine learning",
        **config["default_kwargs"]
    )
    
    if results["success"]:
        # Save results to file (would save to Colab storage)
        report = results["pipeline_results"]
        print(f"Generated {report['experiment_summary']['total_jokes_generated']} jokes")
        print(f"Top score: {report['experiment_summary']['top_score']:.2f}")

def example_save_and_load_results():
    """Example showing how to save and work with results."""
    print("\n=== Save and Load Results Example ===")
    
    joke_search = JokePlanSearchComplete()
    
    # Generate results
    results = joke_search.run_complete_pipeline("technology")
    
    if results["success"]:
        # Save detailed report to JSON
        output_file = "joke_results.json"
        success = joke_search.save_results_to_json(output_file)
        
        if success:
            print(f"Results saved to {output_file}")
            
            # Load and display summary info
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            summary = saved_data["experiment_summary"]
            print(f"Topic: {summary['topic']}")
            print(f"Total jokes: {summary['total_jokes_generated']}")
            print(f"Top score: {summary['top_score']:.2f}")
            
            # Clean up
            os.remove(output_file)
            print("Temporary file cleaned up")

def main():
    """Run all examples."""
    print("JokePlanSearch: Example Usage Demonstrations")
    print("=" * 50)
    
    # Note: Most examples use mock clients since real API keys aren't available
    print("Note: Examples use mock API clients for demonstration.")
    print("For real usage, set your API keys in environment variables.\n")
    
    try:
        example_basic_usage()
        example_quick_demo()
        example_batch_processing()
        example_custom_configuration()
        example_colab_setup()
        example_save_and_load_results()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the individual functions for specific usage patterns.")
        
    except Exception as e:
        print(f"Example execution failed: {str(e)}")
        print("This is expected when using mock clients.")

if __name__ == "__main__":
    main() 