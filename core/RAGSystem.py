import spacy
import chromadb
import uuid
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import SpacyTextSplitter
from core.logger import logger
from core.utils import read_notes

class RAGSystem:
    def __init__(self, memory_path: str = "./data/memory", collection_name: str = "notes"):
        self.memory_path = memory_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.memory_path)
        self.collection = None

        # ✅ Fixed: Proper spaCy model initialization
        self.nlp = self._load_spacy_model()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentence_chunks: List[str] = []

        # ✅ Load and prepare vector database
        chunks = self.load_and_chunk_documents()
        self.setup_vector_db(chunks)
        logger.info("✅ RAGSystem initialized successfully.")

    def _load_spacy_model(self):
        """Load spaCy model or fallback gracefully."""
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully.")
            return nlp
        except OSError:
            logger.warning("⚠️ spaCy model not found. Using basic chunking instead.")
            return None

    def load_and_chunk_documents(self) -> List[str]:
        """Read notes and create sentence-aware text chunks."""
        docs, doc_paths = read_notes()
        if not docs:
            logger.warning("No documents found for ingestion.")
            return []

        # ✅ Use sentence-aware chunking when spaCy is available
        if self.nlp:
            sentence_splitter = SpacyTextSplitter(chunk_size=400, chunk_overlap=50)
            sentence_chunks = []
            for i, doc in enumerate(docs):
                chunks = sentence_splitter.split_text(doc)
                sentence_chunks.extend(chunks)
                logger.info(f"{len(chunks)} chunks created from {doc_paths[i]}")
            logger.info(f"✅ Created {len(sentence_chunks)} total chunks.")
        else:
            logger.warning("spaCy unavailable – using full documents as chunks.")
            sentence_chunks = docs  # fallback: no splitting

        self.sentence_chunks = sentence_chunks
        return sentence_chunks

    def setup_vector_db(self, chunks: List[str] = None):
        """Initialize or update the vector database collection."""
        if not chunks:
            logger.warning("No chunks provided for vector DB setup.")
            return None

        try:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new Chroma collection: {self.collection_name}")
        except Exception:
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing Chroma collection: {self.collection_name}")

        # Skip adding if already populated
        if collection.count() > 0:
            logger.info(f"Collection already contains {collection.count()} chunks.")
            self.collection = collection
            return collection

        # Otherwise, add fresh data
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": "local_notes"} for _ in chunks]
        embeddings = [self.model.encode(chunk).tolist() for chunk in chunks]

        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Stored {len(chunks)} chunks in vector database.")
        self.collection = collection
        return collection

    def process_user_query(self, query: str):
        """Convert user query into an embedding."""
        cleaned_query = query.lower().strip()
        query_embedding = self.model.encode([cleaned_query])[0]
        logger.debug(f"Generated embedding for query: '{query}'")
        return query_embedding

    def search_vector_database(self, query_embedding, top_k: int = 5) -> List[Dict]:
        """Search top-k most relevant documents from the vector DB."""
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

    # --------------------------------------------------------------------------------
    # 6️⃣ Context augmentation
    # --------------------------------------------------------------------------------
    def augment_prompt_with_context(self, query: str, search_results: List[Dict]) -> str:
        """Combine retrieved results and query into an augmented prompt."""
        if not search_results:
            return f"User asked: {query}\n\nNo relevant context found in notes."

        context_parts = [
            f"Source {i+1}: {res['metadata'].get('source', 'Unknown Source')}\n{res['content']}"
            for i, res in enumerate(search_results)
        ]
        context = "\n\n".join(context_parts)

        augmented_prompt = f"""
        Based on the following context, answer the user's question.

        CONTEXT:
        {context}

        QUESTION: {query}

        Please provide a clear, accurate, and concise answer with source references.
        """
        logger.debug(f"Augmented prompt created for query: '{query}'")
        return augmented_prompt

    # --------------------------------------------------------------------------------
    # 7️⃣ Add conversation memory to DB
    # --------------------------------------------------------------------------------
    def add_to_vector_db(self, context: Dict):
        """Add a new conversation or note context into the vector DB."""
        text = context.get("content") or ""
        if not text.strip():
            logger.warning("⚠️ No valid context text found, skipping addition.")
            return

        logger.info(f"Adding conversation context: {text[:100]}...")

        # Split into chunks
        if self.nlp:
            sentence_splitter = SpacyTextSplitter(chunk_size=400, chunk_overlap=50)
            chunks = sentence_splitter.split_text(text)
            logger.info(f"✅ Created {len(chunks)} chunks from conversation context.")
        else:
            chunks = [text]

        # Embed and add
        embeddings = [self.model.encode(chunk).tolist() for chunk in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": "user_conversation"} for _ in chunks]

        collection = self.client.get_collection(self.collection_name)
        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        logger.debug("🧩 Added conversation context to vector DB successfully.")
    def run(self, query: str):
        try:
            query_embedding = self.process_user_query(query)
            search_results = self.search_vector_database(query_embedding)
            response = self.augment_prompt_with_context(query, search_results)
            return {"response": response}
        except Exception as e:
            logger.error(f"Error in RAGSystem.run: {str(e)}")
            return {"response": f"Error: {str(e)}"}