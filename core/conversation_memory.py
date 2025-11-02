import json
from core.logger import logger

class ConversationMemory:
    def __init__(self, path="data/memory.json"):
        self.path = path
        self.messages = []
        self.context = {}
        logger.info("ConversationMemory: Initialized and cleared.")
        # Load existing memory if present
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
                self.messages = data.get("messages", [])
                self.context = data.get("context", {})
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("ConversationMemory: No existing memory found. Starting fresh.")

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def update_context(self, new_context):
        # merge new keys into old context without overwriting everything
        if isinstance(new_context, dict):
            self.context.update(new_context)

    def save(self):
        with open(self.path, "w") as f:
            json.dump({"messages": self.messages, "context": self.context}, f, indent=2)
        logger.info("ConversationMemory: Saved to disk ✅")
