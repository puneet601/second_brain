import os
import google.generativeai as genai
from core.utils import read_notes

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class BaselineBot:
    def __init__(self):
        self.model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        self.context = read_notes()

    def run(self, prompt: str) -> str:
        full_prompt = f"Context:\n{self.context}\n\nUser: {prompt}"
        response = self.model.generate_content(full_prompt)
        return {"response": response.text}
