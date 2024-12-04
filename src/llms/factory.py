from typing import Optional, Dict, Any, Union
from enum import Enum, auto

from .openai_based import ChatOpenAI
from .ollama_based import ChatOllama
from .base import BaseChat


class LLMType(Enum):
    """Enum for supported LLM types"""
    OPENAI = auto()
    OLLAMA = auto()


class LLMFactory:
    """Factory class for creating LLM instances"""

    _client = None
    
    @classmethod
    def register_llm(
        cls,
        llm_type: Union[str, LLMType],
        model: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ):
        """Create an LLM instance based on the specified type.
        
        Args:
            llm_type: Type of LLM to create ("openai", "ollama", or LLMType enum)
            model: Model name/identifier
            **kwargs: Additional configuration parameters
                For OpenAI: api_key, base_url
                For Ollama: host
        
        Returns:
            BaseChat: Configured LLM instance
        
        Raises:
            ValueError: If llm_type is not supported
        """
        if isinstance(llm_type, str):
            try:
                llm_type = LLMType[llm_type.upper()]
            except KeyError:
                raise ValueError(f"Unsupported LLM type: {llm_type}")
        
        if llm_type == LLMType.OPENAI:
            cls._client = ChatOpenAI(
                model=model,
                api_key=kwargs.get('api_key'),
                base_url=kwargs.get('base_url')
            )
        elif llm_type == LLMType.OLLAMA:
            cls._client = ChatOllama(
                model=model,
                host=kwargs.get('host')
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    @classmethod
    def get_llm(cls) -> BaseChat:
        return cls._client

    @classmethod
    def update_config(cls, **kwargs):
        cls._client.config.update_config(kwargs)
