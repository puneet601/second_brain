import asyncio
import json

from core.logger import logger
from core.conversation_memory import ConversationMemory
from core.preference_detector import PreferenceAgent
from core.RAGSystem import RAGSystem
from core.researcher import ResearchAgent
from core.synthesiser import SynthesizerAgent
from core.controller_agent import ControllerAgent
from core.guardrails import redact_pii, detect_pii


class Orchestrator:
    def __init__(self):
        self.memory = ConversationMemory()
        self.rag = RAGSystem()
        self.controller = ControllerAgent()
        self.pref = PreferenceAgent()
        self.research = ResearchAgent(self.rag)
        self.synthesizer_agent = SynthesizerAgent()

    # --------------------------------------------------
    # SYNC CLI ENTRY (wrapper only)
    # --------------------------------------------------
    def run(self):
        print("🤖 Your Second Brain is online. Type 'quit' to exit.\n")

        while True:
            user_input_raw = input("You: ")
            if user_input_raw.strip().lower() == "quit":
                print("Assistant: 👋 Goodbye!")
                return

            asyncio.run(self._run_once_cli(user_input_raw))

    # --------------------------------------------------
    # ASYNC CLI LOGIC
    # --------------------------------------------------
    async def _run_once_cli(self, user_input_raw: str):
        pii_entities = detect_pii(user_input_raw)
        user_input = redact_pii(user_input_raw) if pii_entities else user_input_raw

        decision = await self.controller.decide_action(user_input)
        actions = decision.get("actions", ["chat"])

        print(f"🧩 Controller decided: {json.dumps(decision, indent=2)}")

        for action in actions:
            if action == "research":
                prompt = self.research.retrieve(user_input)
                response = await self.synthesizer_agent.run(prompt)
                print(f"Assistant: {response}\n")

            elif action == "preferences":
                context = await self.pref.run(user_input)  # ✅ FIXED
                logger.info(f"context {context}")

                if context:
                    self.memory.add_message("user", context)
                    self.rag.add_to_vector_db(context)
                    self.memory.save()
                    print("Assistant: 💾 Preference saved to memory.")

            elif action == "chat":
                print(f"Assistant: 💬 General chat: '{user_input}'")

    # --------------------------------------------------
    # ASYNC EVALUATION ENTRY (used by baseline_evals)
    # --------------------------------------------------
    async def run_once(self, user_input_raw: str):
        pii_entities = detect_pii(user_input_raw)
        user_input = redact_pii(user_input_raw) if pii_entities else user_input_raw

        decision = await self.controller.decide_action(user_input)
        actions = decision.get("actions", ["chat"])

        response_text = ""

        for action in actions:
            if action == "research":
                prompt = self.research.retrieve(user_input)
                response_text = await self.synthesizer_agent.run(prompt)

            elif action == "preferences":
                context = await self.pref.run(user_input)  # ✅ FIXED
                logger.info(f"context {context}")

                if context:
                    self.memory.add_message("user", context)
                    self.rag.add_to_vector_db(context)
                    self.memory.save()
                    response_text = "💾 Preference saved to memory."

            elif action == "chat":
                response_text = f"💬 General chat: '{user_input}'"

            elif action == "quit":
                return {"response": "👋 Goodbye!"}

        return {"response": response_text}


if __name__ == "__main__":
    Orchestrator().run()
