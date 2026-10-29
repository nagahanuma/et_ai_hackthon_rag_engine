"""
Decoupled LLM Factory Module.
Supports Groq (Llama-3.3-70B), OpenAI (GPT-4o), and local Ollama.
Provides unified generation and streaming interfaces with temperature=0.0 for zero hallucination.
"""

from typing import Optional, Generator
from abc import ABC, abstractmethod

from groq import Groq
from openai import OpenAI

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class ensuring provider-agnostic interface."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates a complete response."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Yields response tokens as a stream for real-time UI display."""
        pass


class GroqClient(BaseLLMClient):
    """Adapter for ultra-fast, free inference via Groq LPUs."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured in .env")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        logger.info(f"Initialized Groq LLM Client (Model: {self.model}, Temp: {self.temperature})")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content


class OpenAIClient(BaseLLMClient):
    """Adapter for OpenAI models (GPT-4o, GPT-4o-mini)."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, base_url: Optional[str] = None):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        logger.info(f"Initialized OpenAI LLM Client (Model: {self.model})")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content


class LLMFactory:
    """Factory creating the appropriate LLM client based on user preferences or environment settings."""

    @staticmethod
    def get_client(
        custom_groq_key: Optional[str] = None, 
        custom_openai_key: Optional[str] = None,
        custom_model: Optional[str] = None
    ) -> BaseLLMClient:
        # 1. Custom OpenAI Key (BYOK)
        if custom_openai_key and custom_openai_key.strip():
            selected_model = custom_model.strip() if (custom_model and "gpt" in custom_model.lower()) else "gpt-4o-mini"
            logger.info(f"Using User Custom OpenAI Key with model: '{selected_model}'")
            return OpenAIClient(
                api_key=custom_openai_key.strip(),
                model=selected_model,
                temperature=0.0
            )

        # 2. Custom Groq Key (BYOK)
        if custom_groq_key and custom_groq_key.strip():
            selected_model = custom_model.strip() if (custom_model and any(k in custom_model.lower() for k in ["llama", "gpt", "deepseek", "mixtral"])) else settings.LLM_MODEL
            logger.info(f"Using User Custom Groq Key with model: '{selected_model}'")
            return GroqClient(
                api_key=custom_groq_key.strip(),
                model=selected_model,
                temperature=settings.LLM_TEMPERATURE
            )

        # 3. Fallback to System Default Environment Settings
        provider = settings.LLM_PROVIDER.lower().strip()
        if provider == "groq":
            groq_model = custom_model.strip() if (custom_model and any(k in custom_model.lower() for k in ["llama", "gpt", "deepseek", "mixtral"])) else settings.LLM_MODEL
            logger.info(f"Using System Default Groq Key with model: '{groq_model}'")
            return GroqClient(
                api_key=settings.GROQ_API_KEY,
                model=groq_model,
                temperature=settings.LLM_TEMPERATURE
            )
        elif provider == "openai":
            return OpenAIClient(
                api_key=settings.OPENAI_API_KEY,
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE
            )
        elif provider == "ollama":
            return OpenAIClient(
                api_key="ollama",
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                base_url="http://localhost:11434/v1"
            )
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. Supported: 'groq', 'openai', 'ollama'")

