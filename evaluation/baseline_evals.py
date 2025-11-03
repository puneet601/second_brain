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
    {
        "case": Case(
            name="preference_query",
            inputs="I love listening to indie music and jazz.",
            expected_output="Preference stored about user's music taste."
        ),
        "expected_keywords": ["music", "like", "jazz"],
        "expected_actions": ["preferences"]
    },
    {
        "case": Case(
            name="research_task",
            inputs="Find notes related to AI architecture design.",
            expected_output="Notes related to AI architecture design.",
        ),
        "expected_keywords": ["AI", "architecture", "design"],
        "expected_actions": ["research"]
    },
    {
        "case": Case(
            name="quit_command",
            inputs="Exit the program.",
            expected_output="Program terminated.",
        ),
        "expected_keywords": ["exit", "quit"],
        "expected_actions": ["quit"]
    },
]

dataset = Dataset(cases=[c["case"] for c in cases])


# --- Evaluation function for generative models ---
def evaluate_model(model, model_name: str):
    print(f"\n🧠 Evaluating {model_name}...")
    results = []
    total_latency = 0

    for c in cases:
        case = c["case"]
        expected_keywords = c["expected_keywords"]

        start = time.time()
        output = model.run(case.inputs)
        latency = round(time.time() - start, 3)

        # normalize
        if isinstance(output, dict):
            response = output.get("response", "")
        else:
            response = str(output)

        relevance = sum(1 for kw in expected_keywords if kw.lower() in response.lower())
        score = relevance / len(expected_keywords)
        results.append({
            "case": case.name,
            "score": round(score, 2),
            "latency": latency
        })
        total_latency += latency

    avg_score = sum(r["score"] for r in results) / len(results)
    avg_latency = total_latency / len(results)

    print(f"✅ Average Relevance Score: {avg_score:.2f}")
    print(f"⚡ Average Latency: {avg_latency:.2f}s\n")
    return {"model": model_name, "avg_score": avg_score, "avg_latency": avg_latency}


# --- Evaluation for ControllerAgent ---
def evaluate_controller(controller):
    print("\n🧠 Evaluating ControllerAgent...")
    results = []
    total_latency = 0

    for c in cases:
        case = c["case"]
        expected_actions = c.get("expected_actions", [])

        start = time.time()
        decision = controller.decide_action(case.inputs)
        latency = round(time.time() - start, 3)
        total_latency += latency

        actions = decision.get("actions", [])
        # calculate action match score
        matches = sum(1 for act in expected_actions if act in actions)
        score = matches / len(expected_actions) if expected_actions else 0

        print(json.dumps({
            "case": case.name,
            "input": case.inputs,
            "predicted_actions": actions,
            "expected_actions": expected_actions,
            "reason": decision.get("reason", ""),
            "score": round(score, 2),
            "latency": latency
        }, indent=2))

        results.append({"case": case.name, "score": round(score, 2), "latency": latency})

    avg_score = sum(r["score"] for r in results) / len(results)
    avg_latency = total_latency / len(results)
    print(f"\n✅ Controller Accuracy: {avg_score:.2f}")
    print(f"⚡ Average Latency: {avg_latency:.2f}s\n")

    return {"model": "ControllerAgent", "avg_score": avg_score, "avg_latency": avg_latency}


# --- Run evaluation on all systems ---
if __name__ == "__main__":
    summary = []
    summary.append(evaluate_model(baseline, "BaselineBot"))
    summary.append(evaluate_model(rag, "RAGSystem"))
    summary.append(evaluate_controller(controller))

    print("\n📊 Final Comparison Summary:\n")
    for result in summary:
        print(json.dumps(result, indent=2))
