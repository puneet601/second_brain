from core.RAGSystem import RAGSystem
from core.logger import logger

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

class ResearchAgent:
    def __init__(self, rag):
        self.rag = rag

    def retrieve(self, query):
        with tracer.start_as_current_span("ResearchAgent.retrieve"):
            logger.info("ResearchAgent: Searching RAG...")
            query_embedding = self.rag.process_user_query(query)
            search_results = self.rag.search_vector_database(query_embedding)
            prompt = self.rag.augment_prompt_with_context(query, search_results)
            return prompt
