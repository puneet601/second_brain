import uuid
import spacy
from langchain_text_splitters import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional
from core.logger import logger
from core.utils import read_notes

class RAGSystem:
    def __init__(self, client, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentence_chunks: List[str] = []
        self.nlp = self._load_spacy_model()
        self.collection = None

    def _load_spacy_model(self):
        """Load spaCy model or return None if unavailable."""
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded successfully.")
            return nlp
        except OSError:
            logger.warning("⚠️ spaCy model not found, using basic chunking.")
            return None

    def load_and_chunk_documents(self) -> List[str]:
        """Read notes and create sentence-aware text chunks."""
        docs, docPaths = read_notes()
        if not docs:
            logger.warning("⚠️ No documents found for ingestion.")
            return []

        if self.nlp:
            sentence_splitter = SpacyTextSplitter(chunk_size=400, chunk_overlap=50)
            sentence_chunks = []
            for i, doc in enumerate(docs):
                chunks = sentence_splitter.split_text(doc)
                sentence_chunks.extend(chunks)
                logger.info(f"🧩 {len(chunks)} chunks from {docPaths[i]}")
            logger.info(f"✅ Total {len(sentence_chunks)} chunks created.")
        else:
            logger.warning("spaCy unavailable — using full docs as chunks.")
            sentence_chunks = docs

        self.sentence_chunks = sentence_chunks
        return sentence_chunks

    def setup_vector_db(self, chunks: Optional[List[str]] = None):
        """Initialize or update the vector database collection."""
        if chunks is None:
            chunks = self.sentence_chunks

        if not chunks:
            logger.warning("⚠️ No chunks provided for vector DB setup.")
            return None

        try:
            collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"🆕 Created Chroma collection: {self.collection_name}")
        except Exception:
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"📦 Using existing Chroma collection: {self.collection_name}")

        if collection.count() == 0:
            embeddings = [self.model.encode(chunk) for chunk in chunks]  # ✅ Keep as list of float vectors
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{"source": "local_notes"} for _ in chunks]

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,  # ✅ Correct shape
                metadatas=metadatas
            )
            logger.info(f"✅ Stored {len(chunks)} chunks in vector DB.")
        else:
            logger.info(f"ℹ️ Collection already has {collection.count()} chunks.")

        self.collection = collection
        return collection

    def add_to_vector_db(self, context: Dict):
        """Add conversation context dynamically to the vector DB."""
        text = context.get("content") or ""
        if not text.strip():
            logger.warning("⚠️ No valid context text found, skipping addition.")
            return

        logger.info(f"💬 Adding context from conversation: {text[:60]}...")

        # Split into chunks
        if self.nlp:
            sentence_splitter = SpacyTextSplitter(chunk_size=400, chunk_overlap=50)
            chunks = sentence_splitter.split_text(text)
        else:
            chunks = [text]

        embeddings = [self.model.encode(chunk) for chunk in chunks]  # ✅ list of vectors

        collection = self.client.get_collection(self.collection_name)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": "user_conversation"} for _ in chunks]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.debug("🧠 Added conversation context to vector DB successfully.")
