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

This phase is handled by the `generate_combinatorial_plans` method in `joke_generation.py`.

1.  **Selecting Observations**: The system takes the list of primitive observations generated in Phase 1.
2.  **Prompting the LLM**: It then prompts the LLM to find interesting combinations among these observations and formulate them as plans.

    ```python
    # From joke_generation.py
    def generate_combinatorial_plans(self, observations: list, max_plans: int = 15) -> list:
        """Generate joke plans by combining 2-4 primitive observations."""
        # ... logic to sample combinations ...
        for combo in sampled_combos:
            plan_prompt = f"""



**These are the tasks which are meant to be following:**
1. Find a unifying theme or connection between these observations
2. Describe how they could work together in a joke structure
3. Specify the joke format (one-liner, setup-punchline, dialogue, etc.)



3.  **Parsing the Response**: The system parses the LLM's free-text response to extract the plan description.

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