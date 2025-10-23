# llm/completions.py

"""
Centralized logic for calling the LLM (e.g., Vertex AI or OpenAI).
If you add streaming or fallback logic, it can live here.
"""

import asyncio
import logging

from llm.exceptions import LLMCallError
from llm.factory import get_vertex_ai_llm
from llm.token_utils import count_tokens  # (Optional) if you implement token counting

logger = logging.getLogger(__name__)

async def call_vertex_ai(
    prompt: str,
    timeout: int = 15,
    llm=None
) -> str:
    """
    Call Vertex AI with a given prompt. If `llm` is not provided,
    we fetch a default from factory. Raises LLMCallError on failure.

    Args:
        prompt (str): The text prompt to send to Vertex AI.
        timeout (int): Max time in seconds to wait for the LLM response.
        llm: An optional Vertex AI Chat model instance. If not provided,
             we'll call get_vertex_ai_llm() from factory.py.

    Returns:
        str: The text output from the Vertex AI LLM.
    """
    if llm is None:
        llm = get_vertex_ai_llm()  # Fallback to default creation

    try:
        # (Optional) If you have a token counting function:
        token_count = count_tokens(prompt)
        logger.info(f"Prompt token usage: {token_count}")

        # Run the LLM invocation in a background thread to avoid blocking.
        future = asyncio.to_thread(llm.invoke, prompt)
        response = await asyncio.wait_for(future, timeout=timeout)

        # The response object is typically from a LangChain or Vertex AI wrapper.
        # We assume it has a .content attribute with the text response.
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error calling Vertex AI: {e}")
        raise LLMCallError(str(e))
