# evaluation/baseline_evals.py
import json
from pydantic_evals import Case, Dataset
from second_brain.core.controller_agent import ControllerAgent


controller = ControllerAgent()

def task_function(user_input: str) -> str:
    decision = controller.decide_action(user_input)
    return json.dumps(decision["actions"])

cases = [
    Case(
        name="music_preference",
        inputs="What music do I like?",
        expected_output='["preferences", "research"]'
    ),
    Case(
        name="quit_case",
        inputs="Bye!",
        expected_output='["quit"]'
    ),
    # … more cases
]

dataset = Dataset(cases=cases)

# Optionally assign dataset-level evaluators if needed

report = dataset.evaluate_sync(task_function)
print(report.summary())
