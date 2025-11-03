# evaluation/baseline_bot.py
import os
import time

class BaselineBot:
    """
    Baseline retrieval model (no embeddings, no agents, no memory).
    Provides a reference to measure RAG or multi-agent improvements.
    """

    def __init__(self, notes_dir: str = "./notes"):
        self.notes = self._load_notes(notes_dir)

    def _load_notes(self, notes_dir):
        notes = []
        for filename in os.listdir(notes_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(notes_dir, filename), "r") as f:
                    notes.append({
                        "name": filename,
                        "content": f.read()
                    })
        return notes

    def run(self, query: str) -> dict:
        """Search for query in notes, return structured response with metadata."""
        start_time = time.time()
        query_lower = query.lower()
        matches = []

        for note in self.notes:
            if any(word in note["content"].lower() for word in query_lower.split()):
                matches.append(note)

        if not matches:
            return {
                "response": "The notes do not contain any information about that.",
                "matches": [],
                "latency": round(time.time() - start_time, 3)
            }

        response_text = "Based on the notes:\n\n"
        for m in matches[:3]:
            response_text += f"- From **{m['name']}**:\n{m['content'][:300]}...\n\n"

        return {
            "response": response_text.strip(),
            "matches": [m["name"] for m in matches],
            "latency": round(time.time() - start_time, 3)
        }
