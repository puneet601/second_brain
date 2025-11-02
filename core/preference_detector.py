from pydantic_ai import Agent
import json
import re

class PreferenceAgent:
    def __init__(self):
        self.agent = Agent(
            model="google-gla:gemini-2.5-pro",
            instructions="""
            You act as a conversation expert. Your job is to detect if the user shares 
            any information about themselves, their work, projects, or preferences.
            Structure this info as an object with a title based on the data and content discovered.
            If nothing is detected, return an empty string.
            Return ONLY a JSON object in this format:
            {
                "title": "<short descriptive title>",
                "content": "<details>"
                "query_detected":False
            }
            If a preference is found and the user also has a query return 
            {
                "title": "<short descriptive title>",
                "content": "<details>"
                "query_detected": True
            }
            If no prefernce is deteced and user has just asked a question, return an empty object {}.
            """
        )

    def run(self, query: str):
        result = self.agent.run_sync(query)
        print(result)
        output = result.output if hasattr(result, "output") else str(result)

        # remove code block formatting like ```json ... ```
        cleaned = re.sub(r"```(json)?", "", output).strip("` \n")

        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
