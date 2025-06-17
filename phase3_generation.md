# Phase 3: Joke Generation

With a set of creative and structured "combinatorial plans," this phase is where the system finally generates the jokes. This step is more of a "translation" process—translating a high-level plan into a concrete piece of text.

## From Plan to Joke

The core idea of this phase is to use the detailed plan as a highly specific prompt for the LLM. Instead of a vague request like "tell me a joke about coffee," the prompt is now much more constrained and targeted.

This approach ensures that the generated joke adheres to the creative concept developed in the previous phase.

## The Generation Process

This process is handled by the `generate_full_jokes` and `refine_joke` methods within `joke_generation.py`. The system first generates a draft and then refines it.

### Step 1: Initial Joke Generation
The system iterates through each combinatorial plan and generates an initial joke.

1.  **Prompting the LLM**: A prompt is constructed that includes the plan and asks the LLM to execute it.

    ```python
    # From joke_generation.py
    def generate_full_jokes(self, topic: str, plans: list) -> list:
        # ... loops through plans ...
        joke_prompt = f"""
<!-- Topic: \"{topic}\"
Plan: {plan['plan_description']}
Source Observations: {', '.join(plan['observations'])} -->

Execute this plan to create a complete joke. Follow the plan's structure and format exactly.

<!-- Joke: [Your complete joke here]
"""
        response = self.call_llm(joke_prompt)
        extracted_joke = self._extract_joke_from_response(response)
        # ... stores joke ...
    ``` -->

### Step 2: Critique and Refinement
Each generated joke then goes through a critique-and-refine loop to improve its quality.

1. **Critique**: The joke is sent to the LLM with a prompt asking for specific feedback.
2. **Refine**: The original joke, critique, and suggestions are combined into a new prompt asking for an improved version.

    ```python
    # From joke_generation.py
    def refine_joke(self, joke: str, topic: str, plan_description: str) -> str:
        # ... builds critique prompt ...
        critique_response = self.call_llm(critique_prompt)
        
        # ... builds refinement prompt with original joke and critique ...
        refinement_response = self.call_llm(refinement_prompt)
        refined_joke = self._extract_refined_joke(refinement_response)
        return refined_joke.strip()
    ```

## Example

Let's follow one of our plans through this phase.

-   **Input Plan**:
    > "A joke that treats a person's morning coffee ritual with the same seriousness and ceremony as a sacred ancient rite, connecting the 'ritual' and 'need' observations."

-   **Initial Generation Prompt**:
    ```
    Your task is to write a short, funny joke based on the following plan...
    **Joke Plan:** A joke that treats a person's morning coffee ritual...
    **Joke:**
    ```

-   **Potential Generated Joke**:
    > My friend's morning coffee is a whole thing. He says he's 'communing with the bean spirit'. I just think he's addicted.

-   **Refinement Process**:
    -   The system would critique this joke, perhaps noting the punchline is a bit weak.
    -   It would then ask the LLM to improve it, resulting in a potentially funnier and more polished final version.

This refined joke is now ready to be assessed in the next phase: **Joke Evaluation**. 