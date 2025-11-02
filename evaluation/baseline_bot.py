# evaluation/baseline_bot.py
import os

class BaselineBot:
    """
    Simple deterministic baseline for evaluation.
    It searches through plain text notes (no embeddings, no agents).
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

    def run(self, query: str) -> str:
        query_lower = query.lower()
        matches = []

        for note in self.notes:
            if any(word in note["content"].lower() for word in query_lower.split()):
                matches.append(note)

        if not matches:
            return "The notes do not contain any information about that."

        response = "Based on the notes:\n\n"
        for m in matches[:3]:
            response += f"- From **{m['name']}**:\n{m['content'][:300]}...\n\n"

        return response.strip()
