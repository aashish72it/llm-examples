from typing import Optional, List
from langchain_core.language_models import LLM
from pydantic import PrivateAttr
from multimodal.core.llm import LLMClient
import asyncio

class LLMAdapter(LLM):
    """
    Adapter to make LLMClient compatible with ragas/LangChain.
    Provides both sync and async call methods.
    """

    _llm_client: LLMClient = PrivateAttr()

    def __init__(self, llm_client: LLMClient, **kwargs):
        super().__init__(**kwargs)
        self._llm_client = llm_client

    @property
    def _llm_type(self) -> str:
        return "groq-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Synchronous call used by LangChain."""
        messages = [{"role": "user", "content": prompt}]
        return self._llm_client.ask_llm(messages)

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Async call used by ragas/LangChain async flows."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call, prompt, stop)
