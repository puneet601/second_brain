# evaluation/baseline_evals.py
import asyncio
from evaluation.evaluators import compare_baseline_and_orchestrator
from evaluation.baseline_bot import BaselineBot
from core.orchestrator import Orchestrator
import os

def run_eval():
    prompt = "I like Bollywood dances, can you recommend good Bollywood movies?"

    # Load API key from env
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    # Instantiate systems
    baseline = BaselineBot()
    orchestrator = Orchestrator()

    baseline_output = baseline.run(prompt)["response"]
    orchestrator_output = orchestrator.run_once(prompt)["response"]

    result = asyncio.run(
        compare_baseline_and_orchestrator(
            baseline_output=baseline_output,
            orchestrator_output=orchestrator_output,
            prompt=prompt
        )
    )

    print("🔍 Evaluation Result:")
    print(f"Winner: {result['winner']}")
    print(f"Reason: {result['reason']}")

if __name__ == "__main__":
    run_eval()
