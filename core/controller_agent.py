from pydantic_ai import Agent
import json, re
from opentelemetry import trace
tracer = trace.get_tracer(__name__)
class ControllerAgent:
    
    def __init__(self):
        self.agent = Agent(
            model="gemini-2.5-flash",
            instructions="""
            You are an AI Orchestrator that analyzes user input and decides which specialized agents to trigger.

            A single input may contain multiple intents, e.g.:
            - "I like cats and eagles. Can you suggest a good documentary about them?"
              → preferences + research

            Available actions:
            - "research": when the user asks a question to find, recall or wants to know or understand something
            - "preferences": when the user shares a personal fact, habit, or like/dislike
            - "quit": when the user wants to exit

            Return ONLY a JSON object like:
            {
              "actions": ["preferences", "research"],
              "reason": "User expressed a preference and also asked a query"
            }
            """
        )

    def decide_action(self, query: str):
        with tracer.start_as_current_span("ControllerAgent.decide_action"):
            result = self.agent.run_sync(query)
            output = getattr(result, "output", str(result))
            cleaned = re.sub(r"```(json)?", "", output).strip("` \n")
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {"actions": ["chat"], "reason": "Default fallback"}
