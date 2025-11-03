# evaluation/baseline_evals.py
import json
import time
from pydantic_evals import Case, Dataset
from evaluation.baseline_bot import BaselineBot
from core.RAGSystem import RAGSystem
from core.controller_agent import ControllerAgent

# --- Initialize systems ---
baseline = BaselineBot(notes_dir="./notes")
rag = RAGSystem()
controller = ControllerAgent()

# --- Evaluation Dataset ---
cases = [
    Case(
        name="preference_query",
        inputs="What music do I like?",
        expected_output_contains=["music", "like", "favorite"]
    ),
    Case(
        name="personal_memory",
        inputs="Summarize my last journal entry",
        expected_output_contains=["summary", "journal", "entry"]
    ),
    Case(
        name="research_task",
        inputs="Find notes related to AI architecture design",
        expected_output_contains=["AI", "architecture", "design"]
    ),
]

dataset = Dataset(cases=cases)

# --- Evaluation function ---
def evaluate_model(model, model_name: str):
    print(f"\n🧠 Evaluating {model_name}...")
    results = []
    total_latency = 0

    for case in dataset.cases:
        start = time.time()
        output = model.run(case.inputs)
        latency = round(time.time() - start, 3)

        if isinstance(output, dict):
            response = output.get("response", "")
        else:
            response = str(output)

        relevance = sum(1 for kw in case.expected_output_contains if kw.lower() in response.lower())
        score = relevance / len(case.expected_output_contains)
        results.append({"case": case.name, "score": score, "latency": latency})
        total_latency += latency

    avg_score = sum(r["score"] for r in results) / len(results)
    avg_latency = total_latency / len(results)

    print(f"✅ Average Relevance Score: {avg_score:.2f}")
    print(f"⚡ Average Latency: {avg_latency:.2f}s\n")
    return {"model": model_name, "avg_score": avg_score, "avg_latency": avg_latency}

# --- Run evaluation on all systems ---
summary = []
summary.append(evaluate_model(baseline, "BaselineBot"))
summary.append(evaluate_model(rag, "RAGSystem"))
summary.append(evaluate_model(controller, "ControllerAgent"))

print("\n📊 Final Comparison Summary:\n")
for result in summary:
    print(json.dumps(result, indent=2))
