from core.RAGSystem import RAGSystem
from core.logger import logger

class ResearchAgent:
    def __init__(self, rag):
        self.rag = rag

    def retrieve(self, query):
        logger.info("ResearchAgent: Searching RAG...")
        query_embedding = self.rag.process_user_query(query)
        search_results = self.rag.search_vector_database(query_embedding)
        prompt = self.rag.augment_prompt_with_context(query, search_results)
        return prompt
