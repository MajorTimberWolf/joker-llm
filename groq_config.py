#!/usr/bin/env python3
"""
Groq Configuration Management
Optimized configurations for different Groq usage patterns
"""

from joke_plan_search_complete import BiasConfig

class GroqConfig:
    """Configuration management for Groq API usage patterns."""
    
    @staticmethod
    def free_tier_optimized():
        """Ultra-optimized configuration for Groq free tier - minimal API calls."""
        config = BiasConfig()
        config.evaluation_rounds = 1
        config.min_comparisons_per_joke = 1
        
        return {
            "bias_config": config,
            "pipeline_settings": {
                "refinement_rounds": 0,
                "top_n": 3,
                "include_bias_analysis": False,
                "num_angles": 5,  # Reduce from default 11
                "max_jokes": 5    # Generate fewer jokes
            },
            "model_preference": "meta-llama/llama-4-scout-17b-16e-instruct",
            "estimated_calls": 12,
            "estimated_time": "45-60 seconds",
            "rate_limit_delay": 2.1
        }
    
    @staticmethod
    def fast_demo():
        """Quick demo configuration - minimal API calls."""
        config = BiasConfig()
        config.evaluation_rounds = 1
        config.min_comparisons_per_joke = 1
        
        return {
            "bias_config": config,
            "pipeline_settings": {
                "refinement_rounds": 0,
                "top_n": 3,
                "include_bias_analysis": False
            },
            "model_preference": "llama-3.1-8b-instant",  # Fastest
            "estimated_calls": 10,
            "estimated_time": "30-60 seconds"
        }
    
    @staticmethod
    def balanced_quality():
        """Balanced configuration - good quality with reasonable API usage."""
        config = BiasConfig()
        config.evaluation_rounds = 2
        config.min_comparisons_per_joke = 2
        
        return {
            "bias_config": config,
            "pipeline_settings": {
                "refinement_rounds": 1,
                "top_n": 3,
                "include_bias_analysis": False
            },
            "model_preference": "gemma2-9b-it",  # Good balance
            "estimated_calls": 25,
            "estimated_time": "2-3 minutes"
        }
    
    @staticmethod
    def high_quality():
        """High quality configuration - more API calls for better results."""
        config = BiasConfig()
        config.evaluation_rounds = 2
        config.min_comparisons_per_joke = 2
        
        return {
            "bias_config": config,
            "pipeline_settings": {
                "refinement_rounds": 1,
                "top_n": 5,
                "include_bias_analysis": True
            },
            "model_preference": "llama-3.3-70b-versatile",  # Best quality
            "estimated_calls": 40,
            "estimated_time": "4-5 minutes"
        }
    
    @staticmethod
    def rate_limit_safe():
        """Ultra-safe configuration for strict rate limit compliance."""
        config = BiasConfig()
        config.evaluation_rounds = 1
        config.min_comparisons_per_joke = 1
        
        return {
            "bias_config": config,
            "pipeline_settings": {
                "refinement_rounds": 0,
                "top_n": 2,
                "include_bias_analysis": False
            },
            "model_preference": "meta-llama/llama-4-scout-17b-16e-instruct",  # Highest token limit
            "estimated_calls": 8,
            "estimated_time": "20-30 seconds"
        }

class GroqModelSelector:
    """Helper class for selecting optimal Groq models."""
    
    MODELS = {
        "llama-3.1-8b-instant": {
            "tokens_per_min": 6000,
            "requests_per_min": 30,
            "quality": "good",
            "speed": "fastest",
            "use_case": "Quick generation, high volume"
        },
        "gemma2-9b-it": {
            "tokens_per_min": 15000,
            "requests_per_min": 30,
            "quality": "very_good",
            "speed": "fast",
            "use_case": "Balanced quality and speed"
        },
        "llama-3.3-70b-versatile": {
            "tokens_per_min": 12000,
            "requests_per_min": 30,
            "quality": "excellent",
            "speed": "moderate",
            "use_case": "High quality output"
        },
        "meta-llama/llama-4-scout-17b-16e-instruct": {
            "tokens_per_min": 30000,
            "requests_per_min": 30,
            "quality": "excellent",
            "speed": "fast",
            "use_case": "Heavy workloads, high token volume"
        },
        "qwen/qwen3-32b": {
            "tokens_per_min": 6000,
            "requests_per_min": 60,  # Double the request rate!
            "quality": "very_good",
            "speed": "fast",
            "use_case": "High request volume workflows, FREE TIER OPTIMAL"
        }
    }
    
    @classmethod
    def get_best_for_workload(cls, expected_calls: int, priority: str = "balanced"):
        """
        Select the best model for a given workload.
        
        Args:
            expected_calls: Expected number of API calls
            priority: "speed", "quality", "balanced", "token_volume", or "free_tier"
        """
        if priority == "free_tier":
            return "meta-llama/llama-4-scout-17b-16e-instruct"
        elif priority == "speed":
            return "llama-3.1-8b-instant"
        elif priority == "quality":
            return "llama-3.3-70b-versatile"
        elif priority == "token_volume":
            return "meta-llama/llama-4-scout-17b-16e-instruct"
        elif priority == "request_volume":
            return "qwen/qwen3-32b"  # 60 requests/min instead of 30
        else:  # balanced
            if expected_calls > 30:
                return "meta-llama/llama-4-scout-17b-16e-instruct"
            elif expected_calls > 15:
                return "gemma2-9b-it"
            else:
                return "llama-3.3-70b-versatile"
    
    @classmethod
    def show_model_comparison(cls):
        """Display a comparison of available models."""
        print("\n🤖 Groq Model Comparison")
        print("=" * 80)
        print(f"{'Model':<40} {'Tokens/Min':<12} {'Requests/Min':<12} {'Quality':<12} {'Best For'}")
        print("-" * 80)
        
        for model, specs in cls.MODELS.items():
            print(f"{model:<40} {specs['tokens_per_min']:<12} {specs['requests_per_min']:<12} {specs['quality']:<12} {specs['use_case']}")

def get_recommended_config(use_case: str = "balanced"):
    """Get recommended configuration for different use cases."""
    configs = {
        "free": GroqConfig.free_tier_optimized(),
        "demo": GroqConfig.fast_demo(),
        "balanced": GroqConfig.balanced_quality(),
        "quality": GroqConfig.high_quality(),
        "safe": GroqConfig.rate_limit_safe()
    }
    
    return configs.get(use_case, configs["balanced"])

if __name__ == "__main__":
    print("🔧 Groq Configuration Options")
    print("=" * 50)
    
    configs = ["free", "demo", "balanced", "quality", "safe"]
    for config_name in configs:
        config = get_recommended_config(config_name)
        print(f"\n📋 {config_name.upper()} Configuration:")
        print(f"   Model: {config['model_preference']}")
        print(f"   Estimated calls: {config['estimated_calls']}")
        print(f"   Estimated time: {config['estimated_time']}")
    
    GroqModelSelector.show_model_comparison() 