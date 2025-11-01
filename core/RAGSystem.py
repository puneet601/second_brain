from langchain_text_splitters import SpacyTextSplitter
from utils import read_notes
import spacy
from typing import List, Dict, Any
import chromadb

def load_and_chunk_documents()->List[str]:
    try:
        nlp = spacy.load("en_core_web_sm")
        print(" spaCy model loaded successfully")
    except OSError:
        print("⚠️  spaCy model not found. Using basic chunking instead.")
        nlp = None

    if nlp:
        print("-" * 50)
        docs,docPaths = read_notes()
        sentence_splitter = SpacyTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        sentence_chunks = []
        for i, doc in enumerate(docs):
            chunks = sentence_splitter.split_text(doc)
            sentence_chunks.extend(chunks)
            print(f" {len(chunks)} chunks created from {docPaths[i]}")

        print(f"Created {len(sentence_chunks)} chunks:")
    else:
        print("⚠️  spaCy not available - skipping sentence-aware chunking demo")
        print("💡 Install spaCy with: python -m spacy download en_core_web_sm")
    return sentence_chunks


def setup_vector_db(chunks: List[str]):
    client = chromadb.PersistentClient(path="./data/memory")
    try:
        collection = client.create_collection(name="notes",metadata={"hnsw:space": "cosine"})
        print(f"Creating collection {collection.name}")
    except Exception as e:
        collection = client.get_collection("notes")
        print(f"Getting collection {collection.name}")
        
    ids = [str(i) for i in range(len(chunks))]
    documents = chunks
    metadatas = [{"source": "local_notes"} for _ in chunks]
    if collection.count() == 0:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Stored {len(chunks)} chunks in vector database")
    else:
        print(f"Collection already contains {collection.count()} chunks")
    
    print(f"Collection count: {collection.count()}")
    
    return collection


if(__name__=="__main__"):
    chunks = load_and_chunk_documents()
    setup_vector_db(chunks=chunks)
