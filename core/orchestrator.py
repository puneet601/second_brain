import asyncio
import json
from typing import List, Dict, Any

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

    
    async def run(self):
        print(" Your Second Brain is online. Type 'quit' to exit.\n")
        while True:
            user_input_raw = input("You: ")
            if user_input_raw.strip().lower() == "quit":
                print("Assistant: 👋 Goodbye!")
                return
            await self._run_once_cli(user_input_raw)

   
    async def _run_once_cli(self, user_input_raw: str):
        response = await self.run_once(user_input_raw)
        print(f"Assistant: {response['response']}")


    async def run_once(self, user_input_raw: str):
       
        pii_entities = detect_pii(user_input_raw)
        user_input = redact_pii(user_input_raw) if pii_entities else user_input_raw

       
        decision = await self.controller.decide_action(user_input)
        actions = decision.get("actions", ["chat"])

        collected_context: list[dict] = []

       
        for action in actions:
            logger.info(f"[EVAL] Action: {action}")

            if action == "preferences":
                preference = await self.pref.run(user_input)
                if preference:
                    pref_dict = preference.model_dump()
                    collected_context.append(pref_dict)

                    # Persist memory + RAG (same as CLI)
                    self.memory.add_message(role="preference", content=pref_dict)
                    self.rag.add_to_vector_db(pref_dict)
                    self.memory.save()

            else:
                retrieved_context = self.research.retrieve(
                    user_input
                )
                # logger.info(retrieved_context)
                if retrieved_context:
                    collected_context.append({
                        "content": retrieved_context
                    })

       
        final_answer = await self.synthesizer_agent.run(
            user_input,
            collected_context
        )

        return {
            "response": final_answer
        }

if __name__ == "__main__":
    asyncio.run(Orchestrator().run())