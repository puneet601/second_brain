from pydantic_ai import Agent

class SynthesizerAgent:
    def __init__(self):
        self.agent = Agent(
            model="google-gla:gemini-2.5-pro",
            instructions="""You act as a coversation expert.Your job is to answer the user's query in a polished 
            way based on the context provided.If nothing is detected let the user know that the notes and
            previous conversations do not have any information.Limit your response to the knowledge provided in context.
            """
            )

    def generate(self, query, context_results, memory_context):
        result = self.agent.run_sync(query)
        return result.output
