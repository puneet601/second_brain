import json
from core.logger import logger

class ConversationMemory:
    def __init__(self, path="data/memory.json"):
        self.path = path
        self.messages = []
        logger.info("ConversationMemory: Initialized and cleared.")
        
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
                self.messages = data.get("messages", [])
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("ConversationMemory: No existing memory found. Starting fresh.")

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def save(self):
        with open(self.path, "w") as f:
            json.dump({"messages": self.messages}, f, indent=2)
        logger.info("ConversationMemory: Saved to disk ✅")
