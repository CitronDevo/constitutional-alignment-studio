"""
LLM Client Module - Provider-agnostic interface for AI model calls.

This module implements the abstraction layer that allows seamless switching
between OpenAI (GPT-4o) and Ollama (local models) without changing UI code.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Optional, Iterator
import time

import socket

# Force IPv4 to avoid IPv6 blackhole timeouts (especially for Gemini/gRPC)
_original_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(*args, **kwargs):
    responses = _original_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]


socket.getaddrinfo = _ipv4_only_getaddrinfo

from src.models import LLMConfig, CritiqueRewriteResponse


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM implementations must inherit from this class and implement
    the generate() method.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM provider.

        Args:
            config: LLM configuration settings
        """
        self.config = config

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System-level instructions and context
            user_prompt: The specific user task/query

        Returns:
            Generated text response

        Raises:
            ConnectionError: If unable to connect to the LLM service
            ValueError: If the API returns an error
        """
        pass

    def stream_generate(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """
        Generate a streaming response from the LLM (token by token).

        Args:
            system_prompt: System-level instructions and context
            user_prompt: The specific user task/query

        Yields:
            Token strings as they are generated

        Raises:
            ConnectionError: If unable to connect to the LLM service
            ValueError: If the API returns an error
            NotImplementedError: If streaming is not supported by this provider
        """
        # Default implementation: fallback to non-streaming
        response = self.generate(system_prompt, user_prompt)
        yield response

    def parse_critique_rewrite(self, response: str) -> CritiqueRewriteResponse:
        """
        Parse the LLM response to extract critique and rewrite sections.

        Expected format:
        **Critique**: [text here]
        **Rewrite**: [text here]

        Args:
            response: Raw LLM response text

        Returns:
            Structured CritiqueRewriteResponse

        Raises:
            ValueError: If response format is invalid
        """
        # Try to extract using markdown bold markers
        critique_match = re.search(
            r'\*\*Critique\*\*:?\s*(.*?)(?=\*\*Rewrite\*\*|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        rewrite_match = re.search(
            r'\*\*Rewrite\*\*:?\s*(.*?)$',
            response,
            re.DOTALL | re.IGNORECASE
        )

        if not critique_match or not rewrite_match:
            raise ValueError(
                "Could not parse LLM response. Expected format:\n"
                "**Critique**: [text]\n**Rewrite**: [text]\n\n"
                f"Received:\n{response[:200]}..."
            )

        critique = critique_match.group(1).strip()
        rewrite = rewrite_match.group(1).strip()

        return CritiqueRewriteResponse(critique=critique, rewrite=rewrite)


class OpenAIProvider(LLMProvider):
    """
    OpenAI API implementation (GPT-4o, etc.).

    Uses the official OpenAI Python SDK.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI provider.

        Args:
            config: LLM configuration (must include api_key or set OPENAI_API_KEY env var)
        """
        super().__init__(config)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Run: uv add openai"
            )

        # Get API key from config or environment
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide it in LLMConfig."
            )

        self.client = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response using OpenAI API.

        Args:
            system_prompt: System instructions
            user_prompt: User query

        Returns:
            Generated response text

        Raises:
            ConnectionError: If API call fails
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self._create_chat_completion(messages, stream=False)
            content = (response.choices[0].message.content or "").strip()

            # Fallback: if content empty, retry with explicit text response_format
            if not content:
                response = self._create_chat_completion(
                    messages,
                    stream=False,
                    response_format={"type": "text"},
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise ValueError(
                        f"Empty response from model {self.config.model}; "
                        f"usage={getattr(response, 'usage', None)}"
                    )

            return content

        except Exception as e:
            raise ConnectionError(
                f"OpenAI API call failed: {str(e)}"
            ) from e

    def stream_generate(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """
        Generate streaming response using OpenAI API.

        Args:
            system_prompt: System instructions
            user_prompt: User query

        Yields:
            Token strings as they are generated

        Raises:
            ConnectionError: If API call fails
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            stream = self._create_chat_completion(messages, stream=True, timeout=60.0)

            token_count = 0

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        token_count += 1
                        if token_count % 10 == 0:  # Log every 10 tokens
                            print(f"[DEBUG] Received {token_count} tokens so far...")
                        yield delta.content

            print(f"[DEBUG] Streaming completed. Total tokens: {token_count}")

        except Exception as e:
            print(f"[ERROR] OpenAI streaming failed: {str(e)}")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            raise ConnectionError(
                f"OpenAI streaming API call failed: {str(e)}"
            ) from e

    def _create_chat_completion(self, messages, stream: bool = False, **extra_kwargs):
        """
        Helper to call chat.completions with compatibility for max_tokens / max_completion_tokens.
        Tries max_tokens first, falls back to max_completion_tokens if unsupported.
        """
        base_kwargs = dict(
            model=self.config.model,
            messages=messages,
            stream=stream,
        )
        if extra_kwargs:
            base_kwargs.update(extra_kwargs)

        def _call_with_token_key(token_key: str):
            kwargs = dict(base_kwargs)
            kwargs[token_key] = self.config.max_tokens
            return self.client.chat.completions.create(**kwargs)

        try:
            return _call_with_token_key("max_tokens")
        except Exception as e:
            msg = str(e)
            # Retry with max_tokens removed replaced by max_completion_tokens if needed
            if "max_tokens" in msg:
                return _call_with_token_key("max_completion_tokens")
            raise


class OllamaProvider(LLMProvider):
    """
    Ollama local model implementation.

    Connects to Ollama running on localhost:11434.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize Ollama provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)

        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            raise ImportError(
                "Ollama library not installed. Run: uv add ollama"
            )

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """
        Test if Ollama service is running.

        Raises:
            ConnectionError: If Ollama is not accessible
        """
        try:
            # Try to list models to verify connection
            self.ollama.list()
        except Exception as e:
            raise ConnectionError(
                "Cannot connect to Ollama. Please ensure Ollama is running.\n"
                "Start it with: ollama serve\n"
                f"Error: {str(e)}"
            ) from e

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response using Ollama.

        Args:
            system_prompt: System instructions
            user_prompt: User query

        Returns:
            Generated response text

        Raises:
            ConnectionError: If Ollama service is unreachable
        """
        try:
            # Combine system and user prompts for Ollama
            # (Some Ollama models handle system prompts differently)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = self.ollama.generate(
                model=self.config.model,
                prompt=full_prompt,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            )

            return response['response']

        except Exception as e:
            raise ConnectionError(
                f"Ollama generation failed: {str(e)}\n"
                "Make sure the model is pulled: ollama pull {self.config.model}"
            ) from e

    def stream_generate(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """
        Generate streaming response using Ollama.

        Args:
            system_prompt: System instructions
            user_prompt: User query

        Yields:
            Token strings as they are generated

        Raises:
            ConnectionError: If Ollama service is unreachable
        """
        try:
            # Combine system and user prompts for Ollama
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            stream = self.ollama.generate(
                model=self.config.model,
                prompt=full_prompt,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                },
                stream=True
            )

            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']

        except Exception as e:
            raise ConnectionError(
                f"Ollama streaming failed: {str(e)}\n"
                f"Make sure the model is pulled: ollama pull {self.config.model}"
            ) from e


class GeminiProvider(LLMProvider):
    """
    Google Gemini API implementation.

    Uses the google-generativeai library to access Gemini models.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize Gemini provider.

        Args:
            config: LLM configuration

        Raises:
            ImportError: If google-generativeai not installed
            ValueError: If GOOGLE_API_KEY not found
        """
        super().__init__(config)

        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError(
                "Google Generative AI library not installed. Run: uv add google-generativeai"
            )

        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )

        # Configure the library
        self.genai.configure(
            api_key=api_key,
            transport="rest",
        )

        # Initialize the model
        generation_config = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
        }

        self.model = self.genai.GenerativeModel(
            model_name=self.config.model,
            generation_config=generation_config
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response using Gemini API.

        Args:
            system_prompt: System instructions
            user_prompt: User query

        Returns:
            Generated response text

        Raises:
            ConnectionError: If API call fails
        """
        try:
            print(f"[DEBUG] Gemini generate with model: {self.config.model}")
            start = time.perf_counter()

            # Combine system and user prompts
            # Gemini models don't have separate system prompts in the same way
            # as OpenAI, so we prepend the system prompt to the user prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

            response = self.model.generate_content(full_prompt, generation_config={"max_output_tokens": self.config.max_tokens})

            end = time.perf_counter()
            print(f"[DEBUG] Gemini generate completed in {(end-start)*1000:.0f} ms")
            return response.text

        except Exception as e:
            print(f"[ERROR] Gemini generation failed: {str(e)}")
            raise ConnectionError(
                f"Gemini API call failed: {str(e)}"
            ) from e

    def stream_generate(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """
        Generate streaming response using Gemini API.

        Args:
            system_prompt: System instructions
            user_prompt: User query

        Yields:
            Token strings as they are generated

        Raises:
            ConnectionError: If API call fails
        """
        try:
            print(f"[DEBUG] Starting Gemini streaming with model: {self.config.model}")
            print(f"[DEBUG] System prompt length: {len(system_prompt)} chars")
            print(f"[DEBUG] User prompt length: {len(user_prompt)} chars")
            start = time.perf_counter()

            # Combine system and user prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

            response = self.model.generate_content(
                full_prompt,
                stream=True,
                generation_config={"max_output_tokens": self.config.max_tokens},
            )

            print(f"[DEBUG] Stream created successfully, starting iteration...")
            token_count = 0

            for chunk in response:
                # Some chunks contain only finish_reason metadata; skip if no text parts.
                text = None
                try:
                    text = chunk.text  # may raise if no parts
                except Exception:
                    text = None

                if not text and hasattr(chunk, "candidates"):
                    try:
                        for cand in chunk.candidates:  # type: ignore[attr-defined]
                            parts = getattr(getattr(cand, "content", None), "parts", None)
                            if parts:
                                for part in parts:
                                    if getattr(part, "text", None):
                                        text = part.text
                                        break
                                if text:
                                    break
                            # If finish_reason is STOP with no text, skip
                            if getattr(cand, "finish_reason", None) == 1:
                                text = None
                    except Exception:
                        text = None

                if text:
                    token_count += 1
                    if token_count % 10 == 0:  # Log every 10 chunks
                        print(f"[DEBUG] Received {token_count} chunks so far...")
                    yield text

            end = time.perf_counter()
            print(f"[DEBUG] Streaming completed. Total chunks: {token_count}. Elapsed {(end-start)*1000:.0f} ms")

        except Exception as e:
            print(f"[ERROR] Gemini streaming failed: {str(e)}")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            raise ConnectionError(
                f"Gemini streaming API call failed: {str(e)}"
            ) from e


def get_llm_provider(config: LLMConfig) -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider.

    Args:
        config: LLM configuration specifying provider and settings

    Returns:
        Initialized LLMProvider instance

    Raises:
        ValueError: If provider is not recognized
    """
    providers = {
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "gemini": GeminiProvider
    }

    provider_class = providers.get(config.provider)
    if not provider_class:
        raise ValueError(
            f"Unknown provider: {config.provider}. "
            f"Available: {list(providers.keys())}"
        )

    return provider_class(config)


# Testing and demonstration
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("LLM CLIENT DEMONSTRATION")
    print("=" * 70)

    # Test 1: OpenAI Provider (if API key is available)
    print("\n[TEST 1] OpenAI Provider Initialization")
    try:
        openai_config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",  # Use mini for testing
            temperature=0.7
        )
        openai_provider = get_llm_provider(openai_config)
        print("✓ OpenAI provider initialized successfully")

        # Try a simple generation
        if os.getenv("OPENAI_API_KEY"):
            print("\n  Testing generation...")
            response = openai_provider.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'Hello, this is a test!' and nothing else."
            )
            print(f"  Response: {response[:100]}")
        else:
            print("  Skipping generation test (no API key found)")

    except Exception as e:
        print(f"✗ OpenAI provider failed: {e}")

    # Test 2: Ollama Provider (if running)
    print("\n" + "=" * 70)
    print("[TEST 2] Ollama Provider Initialization")
    try:
        ollama_config = LLMConfig(
            provider="ollama",
            model="llama3",
            temperature=0.7
        )
        ollama_provider = get_llm_provider(ollama_config)
        print("✓ Ollama provider initialized successfully")
        print("  (Connection to Ollama service verified)")

    except ConnectionError as e:
        print(f"✗ Ollama not available: {e}")
    except Exception as e:
        print(f"✗ Ollama provider failed: {e}")

    # Test 3: Response Parsing
    print("\n" + "=" * 70)
    print("[TEST 3] Response Parsing")

    sample_response = """
**Critique**: The original response only has 2 sentences and doesn't follow
the ADAPTIVE principle at all. The first letters spell 'SH' instead of 'ADAPTIVE'.

**Rewrite**: Always happy to assist you today. Do you have any specific questions?
All your concerns are important to us. Please tell me what you need. Thank you
for reaching out to us. I'm here to help you. Very glad to assist. Excellent
question you asked.
    """

    try:
        openai_config = LLMConfig(provider="openai")
        provider = OpenAIProvider(openai_config) if os.getenv("OPENAI_API_KEY") else LLMProvider.__new__(LLMProvider)
        provider.__init__ = lambda self, config: None
        provider.__init__(openai_config)

        result = LLMProvider.parse_critique_rewrite(provider, sample_response)
        print("✓ Successfully parsed critique and rewrite")
        print(f"  Critique length: {len(result.critique)} chars")
        print(f"  Rewrite length: {len(result.rewrite)} chars")

    except Exception as e:
        print(f"✗ Parsing failed: {e}")

    print("\n" + "=" * 70)
    print("LLM CLIENT TESTS COMPLETED")
    print("=" * 70)
