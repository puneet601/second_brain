from core.logger import logger
from core.conversation_memory import ConversationMemory
from core.preference_detector import PreferenceAgent
from core.RAGSystem import RAGSystem
from core.researcher import ResearchAgent
from core.synthesiser import SynthesizerAgent
from core.controller_agent import ControllerAgent
import json


class Orchestrator:
    def __init__(self):
        self.memory = ConversationMemory()
        self.rag = RAGSystem()
        self.controller = ControllerAgent()
        self.pref = PreferenceAgent()
        self.research = ResearchAgent(self.rag)
        self.synthesizer_agent = SynthesizerAgent()
        

    def run(self):
        print("🤖 Your Second Brain is online. Type 'quit' to exit.\n")

        while True:
            user_input = input("You: ")

            # Step 1: Decide action
            decision = self.controller.decide_action(user_input)
            actions = decision.get("actions", ["chat"])

            print(f"🧩 Controller decided: {json.dumps(decision, indent=2)}")

        # Step 2: Execute actions 
            for action in actions:
                if action == "research":
                    prompt = self.research.retrieve(user_input)
                    print(f"Assistant: {self.synthesizer_agent.run(prompt)}\n")

                elif action == "preferences":
                    context = self.pref.run(user_input)
                    logger.info(f"context {context}")
                    if context:
                        self.memory.add_message("user", context)
                        self.rag.add_to_vector_db(context)
                        self.memory.save()
                        print("Assistant: 💾 Preference saved to memory.")

                elif action == "quit":
                    print("Assistant: 👋 Goodbye!")
                    return

                elif action == "chat":
                    print(f"Assistant: 💬 General chat: '{user_input}'")

            print()  

# --- Run the orchestrator ---

if __name__ == "__main__":
    app = Orchestrator()
    app.run()
