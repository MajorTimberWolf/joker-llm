#!/usr/bin/env python3
"""JokePlanSearch Groq Runner

This streamlined launcher focuses exclusively on the Groq backend and the
modern `JokePlanSearchPipeline` implementation.  All legacy OpenAI/Anthropic
paths have been removed to simplify the execution flow and avoid
configuration ambiguity.
"""

import os

from joke_plansearch_pipeline import JokePlanSearchPipeline


def main() -> None:
    """Entry-point for executing the PlanSearch pipeline via Groq."""

    # ------------------------------------------------------------------
    # Ensure Groq credentials are configured
    # ------------------------------------------------------------------
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY environment variable not found!")
        print("Please set your Groq API key, e.g.:")
        print("export GROQ_API_KEY='your-api-key-here'")
        return

    # ------------------------------------------------------------------
    # Collect topic from user (with a sensible default for non-interactive
    # environments such as CI).
    # ------------------------------------------------------------------
    try:
        topic = input("Enter a topic for joke generation: ").strip() or "programming"
    except EOFError:
        topic = "programming"

    # ------------------------------------------------------------------
    # Run the full PlanSearch pipeline
    # ------------------------------------------------------------------
    try:
        pipeline = JokePlanSearchPipeline()  # Client created internally using GROQ_API_KEY
        pipeline.run_pipeline(topic)
    except Exception as exc:
        print(f"❌ Error running JokePlanSearch pipeline: {exc}")


if __name__ == "__main__":
    main()
