# Phase 4: Joke Evaluation

Once a set of jokes has been generated, the system needs to determine which ones are good. This phase uses a powerful technique known as **LLM-as-a-Judge**, where a Large Language Model is tasked with evaluating the creative output.

A key focus of this phase is mitigating common biases that can occur when using LLMs for evaluation.

## LLM-as-a-Judge

Instead of relying on complex, hard-coded rules to define what's "funny," we use another LLM to act as an evaluator. The "judge" LLM is given a set of jokes and a rubric, and it returns scores and reasoning.

This approach is flexible and can capture the nuanced, subjective nature of humor far better than traditional metrics.

## The Evaluation Process

This process is handled primarily by the `joke_evaluation.py` module. The core method is `evaluate_jokes`.

1.  **Input**: The method takes the list of generated jokes.

2.  **Bias Mitigation: Shuffling**: Before sending the jokes to the judge, their order is randomized. This is a critical step to mitigate **positional bias**, where models tend to give higher scores to items that appear earlier or later in a list. By shuffling the jokes across multiple evaluation rounds, we can average out this effect.

3.  **Prompting the Judge**: A detailed prompt is sent to the LLM, instructing it to act as a comedy expert. The prompt contains the (shuffled) list of jokes and a clear scoring rubric.

    ```python
    # Simplified from joke_evaluation.py
    def evaluate_jokes(self, jokes: list, topic: str):
        # (Inside a loop for multiple rounds)
        random.shuffle(jokes) # Mitigate positional bias
        
        jokes_for_prompt = "\n".join([f"{i+1}. {joke['joke']}" for i, joke in enumerate(jokes)])

        prompt = f"""
        You are a comedy critic. Your task is to evaluate the following jokes about "{topic}".
        For each joke, provide a score from 1 to 10 for each of the following criteria:
        - **Humor**: How funny is the joke? (1=not funny, 10=hilarious)
        - **Relevance**: How relevant is the joke to the topic? (1=irrelevant, 10=perfectly relevant)
        - **Originality**: How original and surprising is the concept? (1=cliché, 10=highly original)

        Please provide your evaluation in a structured format.

        Jokes to evaluate:
        {jokes_for_prompt}

        Evaluation:
        """
        response = self.call_llm(prompt, temperature=0.2) # Low temperature for consistent judging
        # ... parsing logic ...
    ```
    Note the low `temperature` setting. This is crucial for evaluation tasks to ensure the judge is more objective, consistent, and less prone to random creativity in its assessments.

4.  **Multiple Rounds**: To further improve reliability, the evaluation process is typically run multiple times. In each round, the jokes are re-shuffled. The final score for each joke is the average of its scores across all rounds.

5.  **Parsing Scores**: The system parses the judge's structured response to extract the scores for each joke on each criterion. This data is stored for the final analysis phase.

## Example

-   **Input Jokes (Shuffled List)**:
    1.  `"Why did the programmer quit his job? Because he didn't get arrays."`
    2.  `"My friend takes his morning coffee so seriously. He calls it 'The Great Awakening.'..."`

-   **Judge LLM's Response (Partial)**:
    ```
    Evaluating Joke 1: "Why did the programmer quit his job? Because he didn't get arrays."
    - Humor: 7/10
    - Relevance: 9/10
    - Originality: 5/10
    Reasoning: A classic pun that is highly relevant to programming. It's clever but has been heard before, so originality is moderate.

    ---

    Evaluating Joke 2: "My friend takes his morning coffee so seriously..."
    - Humor: 8/10
    - Relevance: 8/10
    - Originality: 9/10
    Reasoning: An original scenario that builds a funny picture. It connects well to the "coffee culture" theme.
    ```

After collecting these scores, the system proceeds to the final phase: **Analysis and Reporting**. 