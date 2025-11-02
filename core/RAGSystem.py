import spacy
import chromadb
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import SpacyTextSplitter
from core.logger import logger
from core.utils import read_notes
import uuid


class RAGSystem:

    def __init__(self, memory_path: str = "./data/memory", collection_name: str = "notes"):
        self.memory_path = memory_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.memory_path)
        self.collection = None
        self.nlp = self._load_spacy_model()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentence_chunks: List[str] = []
        
        self.setup_vector_db(self,self.load_and_chunk_documents(self))
        logger.info("✅ RAGSystem initialized successfully.")

    @staticmethod
    def _load_spacy_model(self):
        """Load spaCy model or fall back gracefully."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully.")
        except OSError:
            logger.warning("⚠️ spaCy model not found. Using basic chunking instead.")
            self.nlp = None

    @staticmethod
    def load_and_chunk_documents(self) -> List[str]:
        """Read notes and create sentence-aware text chunks."""
        docs, docPaths = read_notes()
        if not docs:
            logger.warning("No documents found for ingestion.")
            return []

        if self.nlp:
            sentence_splitter = SpacyTextSplitter(chunk_size=400, chunk_overlap=50)
            sentence_chunks = []
            for i, doc in enumerate(docs):
                chunks = sentence_splitter.split_text(doc)
                sentence_chunks.extend(chunks)
                logger.info(f"{len(chunks)} chunks created from {docPaths[i]}")
            logger.info(f"✅ Created {len(sentence_chunks)} total chunks.")
        else:
            logger.warning("spaCy unavailable – skipping sentence-aware chunking.")
            sentence_chunks = docs  # fallback: use full docs as chunks

        self.sentence_chunks = sentence_chunks
        return sentence_chunks
    
    @staticmethod
    def setup_vector_db(self, chunks: List[str] = None):
        """Initialize or update the vector database collection."""
        if chunks is None:
            chunks = self.sentence_chunks
        if not chunks:
            logger.warning("No chunks provided for vector DB setup.")
            return None
        

        try:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Creating new Chroma collection: {self.collection_name}")
        except Exception:
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing Chroma collection: {self.collection_name}")

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": "local_notes"} for _ in chunks]

        if collection.count() == 0:
            embeddings = [self.model.encode(chunk).tolist() for chunk in chunks]
            collection.add(ids=ids, documents=chunks,embeddings=embeddings, metadatas=metadatas)
            logger.info(f"Stored {len(chunks)} chunks in vector database.")
        else:
            logger.info(f"Collection already contains {collection.count()} chunks.")

        self.collection = collection
        return collection

    def process_user_query(self, query: str):
        """Encode query text into embeddings."""
        cleaned_query = query.lower().strip()
        query_embedding = self.model.encode([cleaned_query])[0]
        logger.debug(f"Query embedding generated for: '{query}'")
        return query_embedding

    def search_vector_database(self, query_embedding, top_k:int = 5) -> List[Dict]:
        if self.collection is None:
            raise RuntimeError("Vector collection not initialized. Run setup_vector_db() first.")

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        search_results = []
        for i, (doc_id, distance, content, metadata) in enumerate(zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0]
        )):
            similarity = 1 - distance
            search_results.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "similarity": similarity
            })
            logger.debug(f"[{i+1}] Similarity: {similarity:.3f} | Content: {content[:100]}...")

        return search_results

    def augment_prompt_with_context(self, query: str, search_results: List[Dict]) -> str:
        """Combine retrieved results and query into an augmented prompt."""
        if not search_results:
            return f"User asked: {query}\n\nNo relevant context found in notes."

        context_parts = [
            f"Source {i+1}: {res['metadata'].get('source', 'Unknown Source')}\n{res['content']}"
            for i, res in enumerate(search_results)
        ]
        # logger.info(context_parts)
        
        context = "\n\n".join(context_parts)
        augmented_prompt = f"""
        Based on the following context, answer the user's question.

        CONTEXT:
        {context}

        QUESTION: {query}

        Please provide a clear, accurate, and concise answer with source references.
        """
        logger.debug(f"Augmented prompt created for query: '{augmented_prompt}'")
        return augmented_prompt
    

    def add_to_vector_db(self, context:Dict):
        text = context.get("content") or ""
        if not text.strip():
            logger.warning("⚠️ No valid context text found, skipping addition.")
            return
        logger.info(f"context from convo: {text}")
        # Split into chunks
        if self.nlp:
            sentence_splitter = SpacyTextSplitter(chunk_size=400, chunk_overlap=50)
            chunks = sentence_splitter.split_text(text)
            logger.info(f"✅ Created {len(chunks)} chunks from conversation context.")
        else:
            chunks = [text]

        # Embed
        embeddings = [self.model.encode(chunk).tolist() for chunk in chunks]

        # Add to vector DB
        collection = self.client.get_collection(self.collection_name)
        ids = ids = [str(uuid.uuid4()) for _ in chunks]

        metadatas = [{"source": "user_conversation"} for _ in chunks]

        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        logger.debug("🧩 Added conversation context to vector DB successfully.")