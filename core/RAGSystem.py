from langchain_text_splitters import SpacyTextSplitter
from utils import read_notes
import spacy

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

