# Facade Pattern
from llm.factory import LLMFactory
from llm.embeddings import EmbeddingService # type: ignore
from llm.cache.cache_manager import CacheManager
from llm.exceptions import LLMException


class LLMWorkflow:
    def __init__(self, llm_client_type="openai", embedding_model="all-MiniLM-L6-v2", **kwargs):
        """
        Initialize the LLMWorkflow with text generation and embedding capabilities.
        Args:
            llm_client_type (str): Type of LLM client to use ('openai' or 'huggingface').
            embedding_model (str): SentenceTransformer model for embedding generation.
            kwargs: Additional arguments (e.g., API keys).
        """
        try:
            # Initialize text generation client
            self.client = LLMFactory.get_client(client_type=llm_client_type, **kwargs)
            
            # Initialize embedding service
            self.embedder = EmbeddingService(model_name=embedding_model)
            
            # Initialize caching (Redis)
            self.cache = CacheManager(host="localhost", port=6379)
        except Exception as e:
            raise LLMException(f"Failed to initialize LLM Workflow: {str(e)}")

    def generate_embedding(self, text: str) -> dict:
        """
        Generate embeddings for the given text.
        """
        if not text:
            raise ValueError("Input text for embedding generation cannot be empty.")
        
        # Check cache
        cache_key = self.cache.generate_cache_key({"text": text, "task": "embedding"})
        cached_result = self.cache.get_cache(cache_key)
        if cached_result:
            return cached_result

        # Generate embeddings
        embedding = self.embedder.generate_embedding(text)
        
        # Cache the result
        self.cache.set_cache(cache_key, embedding)
        return embedding

    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """
        Summarize the given text using the LLM client.
        """
        if not text:
            raise ValueError("Input text for summarization cannot be empty.")
        
        prompt = f"Summarize the following text: {text}"
        result = self.client.create(prompt=prompt, max_tokens=max_length)["choices"][0]["text"]
        return result.strip()

    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text based on the given prompt.
        """
        if not prompt:
            raise ValueError("Input prompt cannot be empty.")
        
        # Generate text
        result = self.client.create(prompt=prompt, max_tokens=max_length)["choices"][0]["text"]
        return result.strip()
