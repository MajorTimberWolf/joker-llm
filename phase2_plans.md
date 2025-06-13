# Phase 2: Combinatorial Plan Generation

After generating a diverse set of primitive observations, this phase combines them to create "combinatorial plans." These plans are structured concepts that will serve as detailed blueprints for the final jokes.

## What is a Combinatorial Plan?

A combinatorial plan is a high-level joke concept created by merging two or more primitive observations. The goal is to find interesting, surprising, or humorous connections between seemingly disparate ideas.

A plan is **not** a joke. It's the *idea* for a joke.

For example, using the observations about **"programmers"**:

-   **Observation A**: `"They often drink a lot of coffee or energy drinks."`
-   **Observation B**: `"'It works on my machine' is a common excuse."`

A potential **combinatorial plan** could be:
> "Create a joke that connects a programmer's reliance on caffeine with the classic 'it works on my machine' excuse. The punchline should imply the coffee is somehow responsible for the code working locally but not elsewhere."

## Why Generate Plans?

This step is the core of the "PlanSearch" methodology. It formalizes the creative leap of connecting ideas.

1.  **Structured Creativity**: It moves from a divergent process (generating many observations) to a convergent one (combining them into structured ideas). This systematic approach can uncover creative angles that a single-shot prompt might miss.
2.  **Novelty and Surprise**: The most humorous plans often come from combining the most unexpected observations. This process explicitly encourages the search for such connections.
3.  **Guiding the Joke Generation**: A detailed plan gives the LLM a much clearer and more constrained task in the next phase, leading to jokes that are more focused and aligned with a specific creative concept.

## The Generation Process

This phase is handled by the `_generate_combinatorial_plans` method in `JokePlanSearch`.

1.  **Selecting Observations**: The system takes the list of primitive observations generated in Phase 1.
2.  **Prompting the LLM**: It then prompts the LLM to find interesting combinations among these observations and formulate them as plans.

    ```python
    # From joke_plan_search.py
    def _generate_combinatorial_plans(self, observations: List[str], num_plans: int = 10) -> List[str]:
        """Generates combinatorial joke plans from a list of observations."""
        logger.info(f"Generating {num_plans} combinatorial plans.")
        
        observations_str = "\n".join(f"- {obs}" for obs in observations)
        
        prompt = f"""
        Here is a list of observations about a topic:
        {observations_str}

        Your task is to create {num_plans} creative joke plans by combining two or more of these observations.
        A joke plan is a high-level concept or a blueprint for a joke. It is NOT the joke itself.
        The best plans come from combining observations in a surprising or non-obvious way.

        For each plan, describe the core idea and which observations it connects.

        Format the output as a JSON list of strings.
        Example:
        ["A joke about a cat that uses its ability to land on its feet to become a professional stunt double, connecting the 'lands on their feet' and 'independent' observations.", "A concept where a programmer tries to debug their cat, connecting the 'purring' and 'mysterious behavior' observations."]

        Output:
        """
        response_text = self.call_llm(prompt)
        # ... parsing logic ...
    ```

3.  **Parsing the Response**: The system parses the LLM's JSON output to get a list of plan strings.

## Example

-   **Input Observations (for "coffee")**:
    -   `"It's a dark, bitter liquid."`
    -   `"It's a common morning ritual."`
    -   `"It can lead to jitteriness."`
    -   `"People often 'need' it to function."`

-   **Potential Combinatorial Plans**:
    -   `"A joke that treats a person's morning coffee ritual with the same seriousness and ceremony as a sacred ancient rite, connecting the 'ritual' and 'need' observations."`
    -   `"A concept for a superhero whose only power is the jitteriness from drinking too much coffee, but they try to frame it as a useful skill. This connects the 'jitteriness' and 'gives energy' observations."`
    -   `"A joke where someone mistakes a cup of black coffee for a magical potion because of its dark appearance and powerful effects, connecting 'dark liquid' and 'need to function' observations."`

With these plans, the system is ready for the next phase: **Joke Generation**. 