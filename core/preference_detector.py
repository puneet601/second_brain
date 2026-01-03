from pydantic_ai import Agent
from typing import Optional
from core.logger import logger
from core.models.Preference import Preference


class PreferenceAgent:
    def __init__(self):
        self.agent = Agent(
            model="google-gla:gemini-2.5-pro",
            output_type=Optional[Preference],  # ✅ FIXED
            instructions="""
        You are a preference detection agent.

        Your job:
        - Detect whether the user shares information about themselves, their interests, habits, work, or preferences.

        Rules:
        - Return structured data only.
        - Do NOT speak to the user.
        - Do NOT give recommendations.
        - Do NOT explain anything.

        If a preference is found:
        - Populate title and content.
        - Set query_detected = true if the user ALSO asked a question.

        If NO preference is found:
        - Return null.

        Examples:

        User: "I love Bollywood dance movies, can you suggest some?"
        → title: "Bollywood Dance Movies"
        → content: "User enjoys Bollywood movies focused on dance."
        → query_detected: true

        User: "What plants are good for indoor air purification?"
        → null
        """
        )

    async def run(self, user_input: str) -> Optional[Preference]:
        result = await self.agent.run(user_input)

        preference: Optional[Preference] = result.output

        if preference:
            logger.info(f"Preference detected: {preference.model_dump()}")

        return preference
