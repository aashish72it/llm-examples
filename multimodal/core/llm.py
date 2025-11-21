import os
from dotenv import load_dotenv
from groq import Groq
from multimodal.utils.errors import LLMError

class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        try:
            load_dotenv()
            api_key = api_key or os.getenv("GROQ_API_KEY")
            model = model or os.getenv("LLM_MODEL")

            if not api_key:
                raise LLMError("Missing GROQ_API_KEY in environment or .env file")

            self.client = Groq(api_key=api_key)
            self.model = model
        except Exception as e:
            raise LLMError(f"Failed to initialize LLM client: {e}")

    def ask_llm(self, messages: list[dict]) -> str:
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=messages)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}")
