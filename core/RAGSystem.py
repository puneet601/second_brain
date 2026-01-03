import spacy
import chromadb
import uuid
import time
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import SpacyTextSplitter

from core.logger import logger
from core.utils import read_notes
from otel_setup import tracer, agent_latency, rag_hits, agent_invocations


class RAGSystem:
    def __init__(self, memory_path: str = "./data/memory"):
        with tracer.start_as_current_span("rag.init"):
            start = time.time()

            # --- Chroma client ---
            self.client = chromadb.PersistentClient(path=memory_path)

            # --- Collections ---
            self.knowledge_collection = self._get_or_create_collection("knowledge_base")
            self.preference_collection = self._get_or_create_collection("user_preferences")
            self.conversation_collection = self._get_or_create_collection("conversation_memory")

            # --- Models ---
            self.nlp = self._load_spacy_model()
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

            # --- Seed knowledge ---
            self._ingest_seed_notes()

            duration = (time.time() - start) * 1000
            agent_latency.record(duration, {"component": "rag", "stage": "init"})
            agent_invocations.add(1, {"component": "rag"})
            logger.info("✅ RAGSystem initialized successfully")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_or_create_collection(self, name: str):
        try:
            return self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        except Exception:
            logger.info(f"Using existing collection: {name}")
            return self.client.get_collection(name)

    def _load_spacy_model(self):
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, sentence splitting disabled")
            return None

    # ------------------------------------------------------------------
    # Seed knowledge ingestion
    # ------------------------------------------------------------------
    def _ingest_seed_notes(self):
        if self.knowledge_collection.count() > 0:
            logger.info(f"Knowledge base already populated ({self.knowledge_collection.count()} chunks)")
            return

        docs, paths = read_notes()
        if not docs:
            logger.warning("No seed documents found")
            return

        splitter = SpacyTextSplitter(chunk_size=400, chunk_overlap=50) if self.nlp else None

        all_chunks, all_metadata = [], []
        for doc, path in zip(docs, paths):
            chunks = splitter.split_text(doc) if splitter else [doc]
            domain = self._infer_domain(path)
            all_chunks.extend(chunks)
            all_metadata.extend([{"source": "seed", "domain": domain, "path": path} for _ in chunks])
            logger.info(f"{len(chunks)} chunks created from {path}")

        embeddings = self.model.encode(all_chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in all_chunks]

        self.knowledge_collection.add(ids=ids, documents=all_chunks, embeddings=embeddings, metadatas=all_metadata)
        logger.info(f"✅ Stored {len(all_chunks)} knowledge chunks")

    def _infer_domain(self, path: str) -> str:
        p = path.lower()
        if "movie" in p: return "movies"
        if "recipe" in p: return "recipes"
        if "plant" in p: return "plants"
        return "general"

    # ------------------------------------------------------------------
    # Knowledge retrieval
    # ------------------------------------------------------------------
    def retrieve_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        with tracer.start_as_current_span("rag.search"):
            embedding = self.model.encode([query])[0].tolist()
            results = self.knowledge_collection.query(query_embeddings=[embedding], n_results=top_k)

            hits = len(results["documents"][0])
            rag_hits.add(hits)

            docs = []
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                docs.append({"content": doc, "metadata": meta, "similarity": 1 - dist})
            return docs

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------
    def store_preference(self, text: str):
        embedding = self.model.encode([text])[0].tolist()
        self.preference_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"type": "preference", "source": "user"}],
        )

    # ------------------------------------------------------------------
    # Conversation memory
    # ------------------------------------------------------------------
    def add_to_vector_db(self, context: Dict):
        """Add conversation context to conversation_collection safely."""
        text = context.get("content") or ""
        if not text.strip():
            logger.warning("⚠️ No valid context text found, skipping addition.")
            return

        logger.info(f"Adding conversation context: {text[:100]}...")

        # Chunk text
        chunks = [text]
        if self.nlp:
            splitter = SpacyTextSplitter(chunk_size=400, chunk_overlap=50)
            chunks = splitter.split_text(text)
            logger.info(f"✅ Created {len(chunks)} chunks from conversation context.")

        embeddings = [self.model.encode(chunk).tolist() for chunk in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": "user_conversation"} for _ in chunks]

        self.conversation_collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        logger.debug("🧩 Added conversation context to vector DB successfully.")

    def store_conversation(self, text: str):
        self.conversation_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            metadatas=[{"type": "conversation", "source": "user"}],
        )
