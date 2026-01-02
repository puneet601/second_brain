import json
import os
import asyncio
from pydantic_evals.evaluators import LLMJudge
from core.utils import read_notes

async def compare_baseline_and_orchestrator(baseline_output: str, orchestrator_output: str, prompt: str):
    """
    Compare BaselineBot and Orchestrator outputs using Gemini as judge,
    limited strictly to context from notes.
    """

    # Ensure API key is set in the environment
    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("⚠️ GEMINI_API_KEY not set in environment.")

    # The model reads the key automatically from env — no api_key arg needed
    judge = LLMJudge(model="gemini-2.5-pro",rubric='''Score the response on a scale of 1–5 for:
    1. Faithfulness to provided context
    2. Helpfulness and completeness
    3. Appropriateness to user intent
    4. Absence of hallucinations
    Prefer the system with higher overall score.''')

    notes_context = read_notes() or "The notes and previous conversations do not contain any relevant information."

    comparison_prompt = comparison_prompt = f"""
        You are an impartial evaluator comparing two AI systems.

        User prompt:
        "{prompt}"

        System context (this is the *only* allowed source of truth):
        {notes_context}

        BaselineBot response:
        {baseline_output}

        Orchestrator response:
        {orchestrator_output}

        Evaluate which system:
        - Uses context from notes accurately
        - Avoids hallucinating information not found in notes
        - Feels more helpful and natural

        Respond ONLY in strict JSON:
        {{
        "winner": "baseline" | "orchestrator" | "tie",
        "reason": "<short explanation>",
        "scores": {{
            "relevance": 1,
            "helpfulness": 1,
            "intent_alignment": 1
        }}
        }}
        """


    # Call the judge asynchronously
    raw_result = await judge.evaluate(comparison_prompt)

    # Safely parse JSON
    try:
        result = json.loads(raw_result)
    except Exception:
        result = {"winner": "tie", "reason": f"Judge output not valid JSON: {raw_result}"}

    return result