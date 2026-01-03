from core.RAGSystem import RAGSystem
from core.logger import logger
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class ResearchAgent:
    """
    Responsible for retrieving grounded knowledge from the RAG system.
    Only queries the 'knowledge_base' collection.
    """
    def __init__(self, rag: RAGSystem):
        self.rag = rag

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve the top-k knowledge chunks relevant to the user's query.
        Ignores preferences and conversation memory to prevent hallucinations.
        """
        if not query or not isinstance(query, str):
            return []

        with tracer.start_as_current_span("ResearchAgent.retrieve"):
            logger.info("ResearchAgent: Searching knowledge base...")

            try:
                results = self.rag.retrieve_knowledge(query=query, top_k=top_k)
                if not results:
                    logger.info("No relevant knowledge found for query")
                else:
                    logger.info(f"Found {len(results)} relevant knowledge chunks")
                return results

            except Exception as e:
                logger.exception("ResearchAgent failed during retrieval")
                return []
