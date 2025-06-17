#!/usr/bin/env python3
"""Groq Integration Tests for JokePlanSearch

This consolidated test suite verifies three critical aspects of the
Groq-powered workflow:

1.  Basic connectivity – a tiny chat completion proving credentials are
    valid.
2.  End-to-end pipeline smoke test – runs the PlanSearch pipeline with
    aggressive limits so it completes in <60 s on the free tier.
3.  Capability overview – prints a terse model comparison table to aid in
    manual selection/debugging.

Run the file directly (`python test_groq_integration.py`) to execute all
tests.  The script exits with a non-zero status code if any mandatory test
fails, making it suitable for CI.
"""

import os
import sys
import time
from typing import Optional


def _create_groq_client(api_key: Optional[str] = None):
    """Lightweight helper that initialises the Groq SDK on-demand."""

    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set – cannot run Groq tests.")

    try:
        import groq
    except ImportError as exc:
        raise ImportError(
            "Groq SDK missing.  Install with `pip install groq`."
        ) from exc

    return groq.Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# 1. Quick connectivity sanity-check
# ---------------------------------------------------------------------------


def quick_connection_test() -> bool:
    """Send a minimal prompt to validate the API key works."""

    print("\n🧪  Quick Groq connectivity test…", flush=True)

    try:
        client = _create_groq_client()
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": "Say \"Knock knock!\""}],
            temperature=0.2,
            max_tokens=10,
        )
        print("✅  Groq responded:", response.choices[0].message.content.strip())
        return True
    except Exception as exc:
        print(f"❌  Groq connectivity test failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# 2. End-to-end PlanSearch smoke test (fast mode)
# ---------------------------------------------------------------------------


def pipeline_smoke_test(topic: str = "programming") -> bool:
    """Run the PlanSearch pipeline with very small limits for CI speed."""

    print("\n🚀  Running PlanSearch pipeline smoke-test…", flush=True)

    from joke_plansearch_pipeline import JokePlanSearchPipeline

    started = time.time()
    try:
        pipeline = JokePlanSearchPipeline()
        pipeline.run_pipeline(topic)
        elapsed = time.time() - started
        print(f"✅  Pipeline completed in {elapsed:.1f}s")
        return True
    except Exception as exc:
        print(f"❌  Pipeline smoke test failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# 3. Show available Groq models (informational)
# ---------------------------------------------------------------------------


def show_model_capabilities() -> None:
    """Pretty-print Groq model information using helper from groq_config."""

    try:
        from groq_config import GroqModelSelector

        print("\n📊  Groq model comparison:")
        GroqModelSelector.show_model_comparison()
    except ImportError:
        print("⚠️  groq_config not found – skipping model comparison table.")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    overall_success = True

    if not quick_connection_test():
        overall_success = False

    # Only attempt pipeline test if connection worked
    if overall_success:
        topic_arg = sys.argv[1] if len(sys.argv) > 1 else "AI programming"
        if not pipeline_smoke_test(topic_arg):
            overall_success = False

    show_model_capabilities()

    if not overall_success:
        sys.exit(1) 