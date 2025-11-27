"""
Pydantic data models for the Constitutional AI Adaptive Tool.

These models serve as the "source of truth" for data validation and structure.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ConversationTurn(BaseModel):
    """
    Represents a single conversation turn from dev.jsonl.

    Expected format:
    {"user": "...", "assistant": "..."} or {"user": "...", "bot": "..."}
    """

    user: str = Field(..., description="User's input message")
    assistant: Optional[str] = Field(None, description="Assistant's response")
    bot: Optional[str] = Field(None, description="Bot's response (alternative to assistant)")

    @field_validator("user")
    @classmethod
    def validate_user_not_empty(cls, v: str) -> str:
        """Ensure user message is not empty."""
        if not v or not v.strip():
            raise ValueError("User message cannot be empty")
        return v.strip()

    def model_post_init(self, __context):
        """Normalize bot/assistant field after initialization."""
        # If bot is provided but not assistant, use bot as assistant
        if self.bot and not self.assistant:
            self.assistant = self.bot
        # Ensure we have an assistant response
        if not self.assistant or not self.assistant.strip():
            raise ValueError("Assistant/bot response cannot be empty")
        # Clean up
        self.assistant = self.assistant.strip()
        # Remove bot field to keep model clean
        self.bot = None


class ConstitutionalExample(BaseModel):
    """
    A constitutional example: original response + critique + rewrite.

    This is what users create in the UI to teach the LLM.
    """

    user_prompt: str = Field(
        ...,
        description="The original query from the user (Human) that elicited the response"
    )
    source_id: int = Field(
        ...,
        ge=0,
        description="Line number in dev.jsonl (0-indexed) that this example came from"
    )
    principle: str = Field(
        default="ADAPTIVE",
        description="The constitutional principle being applied"
    )
    original_response: str = Field(
        ...,
        description="The original (flawed) assistant response"
    )
    critique: str = Field(
        ...,
        description="Explanation of why the response failed the principle"
    )
    rewrite: str = Field(
        ...,
        description="The corrected response that satisfies the principle"
    )

    # Metadata for tracking provenance
    model_used: Optional[str] = Field(
        default=None,
        description="Teacher model used to generate critique and rewrite (e.g., 'gpt-4o', 'llama3.2')"
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp when this example was created"
    )
    active: bool = Field(
        default=True,
        description="If false, this example is ignored in few-shot context"
    )
    version_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this saved version (history/audit log)"
    )

    @field_validator("original_response", "critique", "rewrite")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Ensure all fields have content."""
        if not v or not v.strip():
            raise ValueError("Constitutional example fields cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
            "principle": "ADAPTIVE",
            "user_prompt": "Hello! How's it going?",
            "original_response": "Hello! How's it going?",
            "critique": "The response uses only 2 sentences, but we need exactly 8 sentences where the first letters spell ADAPTIVE.",
            "rewrite": "Always happy to help. Do you have questions? All your concerns matter. Please tell me more. Thank you for reaching out. I'm here to assist. Very glad you asked. Excellent question indeed.",
            "source_id": 0
        }
    }


class TestResult(BaseModel):
    """
    Result of evaluating zero-shot vs few-shot outputs.
    """

    task_id: int = Field(
        ...,
        ge=0,
        description="Index of the task in dev.jsonl"
    )
    zero_shot_output: str = Field(
        ...,
        description="Model output without any constitutional examples"
    )
    few_shot_output: str = Field(
        ...,
        description="Model output using the saved constitutional examples"
    )
    is_improved: bool = Field(
        ...,
        description="Whether the few-shot output improved over zero-shot"
    )

    @field_validator("zero_shot_output", "few_shot_output")
    @classmethod
    def validate_output_not_empty(cls, v: str) -> str:
        """Ensure outputs are not empty."""
        if not v or not v.strip():
            raise ValueError("Model outputs cannot be empty")
        return v.strip()


class LLMConfig(BaseModel):
    """
    Configuration for LLM provider selection and settings.
    """

    provider: Literal["openai", "ollama", "gemini"] = Field(
        default="ollama",
        description="Which LLM provider to use"
    )
    model: str = Field(
        default="llama3.2",
        description="Model name (e.g., 'gpt-4o' for OpenAI, 'llama3.2' for Ollama)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation"
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Maximum tokens to generate"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI (read from env if not provided)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }


class CritiqueRewriteResponse(BaseModel):
    """
    Structured response from the teacher LLM.

    The LLM should return both a critique and a rewrite.
    """

    critique: str = Field(..., description="Critique of the original response")
    rewrite: str = Field(..., description="Rewritten response following ADAPTIVE")

    @field_validator("critique", "rewrite")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Ensure both fields have content."""
        if not v or not v.strip():
            raise ValueError("Critique and rewrite must not be empty")
        return v.strip()
