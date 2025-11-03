import json
import os
import asyncio
from pydantic_evals.evaluators.llm_as_a_judge import LLMJudge
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
    judge = LLMJudge(model="gemini-2.5-pro",rubric='All factual claims in the response are supported by the provided context')

    notes_context = read_notes() or "The notes and previous conversations do not contain any relevant information."

    comparison_prompt = f"""
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
      "reason": "<short explanation>"
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