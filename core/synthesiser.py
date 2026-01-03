from core.logger import logger
from pydantic_ai import Agent
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class SynthesizerAgent:
    """
    Synthesizes user queries and retrieved context into rich, detailed, actionable answers.
    """
    def __init__(self):
        self.agent = Agent(
            model="google-gla:gemini-2.5-pro",
            instructions="""
            You are a helpful assistant that answers questions using the context provided. 
            Your answers must be:
            - Rich and detailed
            - Structured with lists or bullets where appropriate
            - Include descriptions, ratings, or key highlights for items like movies, recipes, or plants
            - Avoid repeating information
            - Cite the source (from the CONTEXT) for each fact or item
            - Follow english grammar rules and do not use unecessary '*' or symbols
            - The final answer must be conversational 
            Please provide a response in plain text optimised to be easily readable.
only use bullets, numbers in a list and do not use markdown formatting. 
Give clear, readable sentences only.
            - do not mention source as source 1 or 2.


        
            """
        )

    async def run(self, query: str, context_chunks: list[dict]):
        with tracer.start_as_current_span("SynthesizerAgent.run"):
            # Combine context into a single text block for the prompt
            context_text = "\n\n".join(
                f"Source {i+1}: {chunk.get('metadata', {}).get('source', 'Unknown')}\n{chunk['content']}"
                for i, chunk in enumerate(context_chunks)
            )

            prompt = f"""
            Based on the following context, provide a detailed, structured answer to the user's question.

            CONTEXT:
            {context_text}

            QUESTION: {query}

            Please provide a clear, rich, and informative response to all the queries asked in the question in a structured format, including highlights, ratings, or descriptions where applicable, and cite sources from the context.
            answer when user asks a query, if the user only tells something acknowledge it with grace.
            """

            try:
                result = await self.agent.run(prompt)
                answer = result.output.strip()
                logger.info("SynthesizerAgent: Generated response successfully")
                return answer
            except Exception as e:
                logger.exception(f"SynthesizerAgent failed: {str(e)}")
                return "Sorry, I couldn't generate a detailed answer at this time."
