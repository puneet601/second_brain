import asyncio
import json
from typing import Literal, Dict, Any

from pydantic import BaseModel
from pydantic_ai import Agent

from evaluation.baseline_bot import BaselineBot
from core.orchestrator import Orchestrator
from evaluation.test_cases import TEST_CASES


# -------------------------
# Judge output schema
# -------------------------
class ComparisonResult(BaseModel):
    winner: Literal["baseline", "orchestrator", "tie"]
    reason: str


# -------------------------
# Judge agent
# -------------------------
judge_agent = Agent(
    model="google-gla:gemini-2.5-pro",
    output_type=ComparisonResult,
    instructions="""
You are an impartial evaluator comparing two assistant responses.

Evaluate based on:
- Correctness
- Completeness
- Relevance to the prompt
- Clarity of explanation

Choose:
- "baseline" if the baseline answer is better
- "orchestrator" if the orchestrator answer is better
- "tie" if both are comparable

Always explain your reasoning clearly.
""",
)


# -------------------------
# Single evaluation
# -------------------------
async def evaluate_case(
    prompt: str,
    baseline_output: str,
    orchestrator_output: str,
) -> ComparisonResult:
    judge_prompt = f"""
User prompt:
{prompt}

Baseline response:
{baseline_output}

Orchestrator response:
{orchestrator_output}

Decide which response is better and why.
"""
    result = await judge_agent.run(judge_prompt)
    return result.output


# -------------------------
# Main evaluation loop
# -------------------------
async def run_eval() -> Dict[str, Any]:
    baseline = BaselineBot()
    orchestrator = Orchestrator()

    results = []

    for case in TEST_CASES:
        prompt = case["prompt"]

        # Baseline is sync-only (OK)
        baseline_output = baseline.run(prompt)["response"]

        # Orchestrator is async-only (MUST await)
        orchestrator_result = await orchestrator.run_once(prompt)
        orchestrator_output = orchestrator_result["response"]

        comparison = await evaluate_case(
            prompt=prompt,
            baseline_output=baseline_output,
            orchestrator_output=orchestrator_output,
        )

        results.append(
            {
                "id": case["id"],
                "prompt": prompt,
                "winner": comparison.winner,
                "reason": comparison.reason,
            }
        )

    orchestrator_wins = sum(1 for r in results if r["winner"] == "orchestrator")
    baseline_wins = sum(1 for r in results if r["winner"] == "baseline")
    ties = sum(1 for r in results if r["winner"] == "tie")

    summary = {
        "total_cases": len(results),
        "orchestrator_wins": orchestrator_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "win_rate": orchestrator_wins / len(results) if results else 0.0,
    }

    print("\n📊 Evaluation Summary")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Orchestrator wins: {summary['orchestrator_wins']}")
    print(f"Baseline wins: {summary['baseline_wins']}")
    print(f"Ties: {summary['ties']}")
    print(f"Win rate: {summary['win_rate']:.2%}")

    with open("evaluation_results.json", "w") as f:
        json.dump(
            {"summary": summary, "results": results},
            f,
            indent=2,
        )

    return {"summary": summary, "results": results}


# -------------------------
# Entry point (ONLY place asyncio.run is allowed)
# -------------------------
if __name__ == "__main__":
    asyncio.run(run_eval())
