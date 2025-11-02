from pydantic_ai import Agent
import json, re

class ControllerAgent:
    def __init__(self):
        self.agent = Agent(
            model="google-gla:gemini-2.5-pro",
            instructions="""
            You are an AI Orchestrator that analyzes user input and decides which specialized agents to trigger.

            A single input may contain **multiple intents**, e.g.:
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
        result = self.agent.run_sync(query)
        output = getattr(result, "output", str(result))
        cleaned = re.sub(r"```(json)?", "", output).strip("` \n")
        print(f"controller output {result}")
        print(f"controller output cleaned {cleaned}")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"actions": ["chat"], "reason": "Default fallback"}
