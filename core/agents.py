from core.conversation_memory import ConversationMemory
from core.preference_detector import PreferenceAgent
from core.RAGSystem import RAGSystem
from core.researcher import ResearchAgent
from core.synthesiser import SynthesizerAgent

class SecondBrainApp:
    def __init__(self):
        print("Initializing SecondBrainApp...")
        self.memory = ConversationMemory()
        self.preference_agent = PreferenceAgent()
        self.rag = RAGSystem()
        self.research_agent = ResearchAgent(self.rag)
        self.synthesizer_agent = SynthesizerAgent()
        

    def handle_user_input(self, user_input: str):
        self.memory.add_message("user", user_input)
        context = self.preference_agent.analyze(user_input)
        if context:
            self.memory.update_context(context)
            
        search_results = self.research_agent.retrieve(user_input)

        response = self.synthesizer_agent.generate(user_input, search_results, self.memory.context)

        self.memory.add_message("assistant", response)
        self.memory.save()
        return response

if __name__ == "__main__":
    app = SecondBrainApp()
    print("\n🤖 Your Second Brain is ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("🧠 Goodbye! Saving memory...")
            app.memory.save()
            break
        response = app.handle_user_input(user_input)
        print(f"Assistant: {response}\n")
