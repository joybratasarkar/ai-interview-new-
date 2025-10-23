# llm/exceptions.py

"""
Custom exception classes for LLM-related errors and retrieval errors.
"""

class LLMCallError(Exception):
    """
    Raised when an error occurs while calling the LLM API
    (e.g., network issues, timeouts, or provider errors).
    """
    pass


class RetrievalError(Exception):
    """
    Raised when an error occurs during retrieval from
    external sources (databases, APIs, etc.).
    """
    pass


class QueryClassificationError(Exception):
    """
    Raised when an error occurs while classifying the user's query
    into different retrieval or pipeline strategies.
    """
    pass
