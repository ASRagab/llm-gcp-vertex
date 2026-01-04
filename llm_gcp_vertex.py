"""
LLM plugin for Google Cloud Vertex AI.

Supports Vertex AI Gemini models and Model Garden partner models (Claude).
Authentication via Application Default Credentials (ADC).
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import AsyncGenerator, Callable, Generator
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import llm
from pydantic import Field, field_validator

if TYPE_CHECKING:
    from anthropic import AnthropicVertex, AsyncAnthropicVertex
    from anthropic.types import MessageParam
    from google.genai import Client as GenAIClient


def get_project_id() -> str:
    """
    Get GCP project ID from LLM keys or environment variable.

    Priority:
        1. LLM key store (llm keys set vertex-project)
        2. LLM_VERTEX_CLOUD_PROJECT environment variable
        3. GOOGLE_CLOUD_PROJECT environment variable (backwards compatible)
    """
    project = llm.get_key(key_alias="vertex-project")
    if project:
        return project

    project = os.environ.get("LLM_VERTEX_CLOUD_PROJECT")
    if project:
        return project

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        return project

    raise ValueError(
        "\n".join(
            (
                "Vertex AI project ID required. Set via:",
                "  llm keys set vertex-project",
                "  or LLM_VERTEX_CLOUD_PROJECT environment variable",
                "  (fallback: GOOGLE_CLOUD_PROJECT)",
            )
        )
    )


def get_location() -> str:
    """
    Get GCP location from LLM keys or environment variable.

    Priority:
        1. LLM key store (llm keys set vertex-location)
        2. LLM_VERTEX_CLOUD_LOCATION environment variable
        3. GOOGLE_CLOUD_LOCATION environment variable (backwards compatible)
        4. Default: us-central1
    """
    location = llm.get_key(key_alias="vertex-location")
    if location:
        return location

    location = os.environ.get("LLM_VERTEX_CLOUD_LOCATION")
    if location:
        return location

    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if location:
        return location

    return "us-central1"


@lru_cache(maxsize=1)
def get_genai_client(
    project: str | None = None, location: str | None = None
) -> GenAIClient:
    """
    Get a cached google.genai client configured for Vertex AI.

    Uses Application Default Credentials for authentication.
    Client is cached after first call.
    """
    from google import genai

    return genai.Client(
        vertexai=True,
        project=project or get_project_id(),
        location=location or get_location(),
    )


class GeminiOptions(llm.Options):
    """Generation options for Gemini models."""

    temperature: float | None = Field(
        default=None,
        description="Controls randomness. Lower is more deterministic.",
        ge=0.0,
        le=2.0,
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens to generate.",
        ge=1,
    )
    top_p: float | None = Field(
        default=None,
        description="Nucleus sampling threshold.",
        ge=0.0,
        le=1.0,
    )
    top_k: int | None = Field(
        default=None,
        description="Top-k sampling parameter.",
        ge=1,
    )


def build_config(prompt: llm.Prompt) -> Any:
    """Build generation config from prompt options."""
    from google.genai.types import GenerateContentConfig

    opts = cast(GeminiOptions, prompt.options)
    temperature = opts.temperature
    max_output_tokens = opts.max_output_tokens
    top_p = opts.top_p
    top_k = opts.top_k
    system_instruction = prompt.system

    if (
        temperature is None
        and max_output_tokens is None
        and top_p is None
        and top_k is None
        and system_instruction is None
    ):
        return None

    return GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        top_k=top_k,
        system_instruction=system_instruction,
    )


def build_contents(
    prompt: llm.Prompt, conversation: llm.Conversation | None
) -> list[object]:
    """Build contents list from prompt and synchronous conversation history."""
    from google.genai.types import Content, Part

    contents: list[object] = []

    if conversation:
        for prev_response in conversation.responses:
            prev = cast(llm.Response, prev_response)
            contents.append(Content(role="user", parts=[Part(text=prev.prompt.prompt)]))
            contents.append(Content(role="model", parts=[Part(text=prev.text())]))

    contents.append(Content(role="user", parts=[Part(text=prompt.prompt)]))

    return contents


async def build_contents_async(
    prompt: llm.Prompt, conversation: llm.AsyncConversation | None
) -> list[object]:
    """Build contents list from prompt and asynchronous conversation history."""
    from google.genai.types import Content, Part

    contents: list[object] = []

    if conversation:
        for prev_response in conversation.responses:
            prev = cast(llm.AsyncResponse, prev_response)
            contents.append(Content(role="user", parts=[Part(text=prev.prompt.prompt)]))
            contents.append(Content(role="model", parts=[Part(text=await prev.text())]))

    contents.append(Content(role="user", parts=[Part(text=prompt.prompt)]))

    return contents


class VertexGeminiModel(llm.Model):
    """Synchronous Vertex AI Gemini model."""

    model_id: str
    vertex_model_name: str
    can_stream: bool = True
    needs_key: str | None = None

    class Options(GeminiOptions): ...  # type: ignore[override]

    def __init__(self, model_id: str, vertex_model_name: str):
        self.model_id = model_id
        self.vertex_model_name = vertex_model_name

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None = None,
    ) -> Generator[str, None, None]:
        client = get_genai_client()
        config = build_config(prompt)
        contents = build_contents(prompt, conversation)

        if stream:
            for chunk in client.models.generate_content_stream(
                model=self.vertex_model_name,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    yield chunk.text
        else:
            result = client.models.generate_content(
                model=self.vertex_model_name,
                contents=contents,
                config=config,
            )
            if result.text:
                yield result.text


class AsyncVertexGeminiModel(llm.AsyncModel):
    """Asynchronous Vertex AI Gemini model."""

    model_id: str
    vertex_model_name: str
    can_stream: bool = True
    needs_key: str | None = None

    class Options(GeminiOptions): ...  # type: ignore[override]

    def __init__(self, model_id: str, vertex_model_name: str):
        self.model_id = model_id
        self.vertex_model_name = vertex_model_name

    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.AsyncResponse,
        conversation: llm.AsyncConversation | None = None,
    ) -> AsyncGenerator[str, None]:
        client = get_genai_client()
        config = build_config(prompt)
        contents = await build_contents_async(prompt, conversation)

        if stream:
            stream_iter = client.aio.models.generate_content_stream(
                model=self.vertex_model_name,
                contents=contents,
                config=config,
            )
            if inspect.isawaitable(stream_iter):
                stream_iter = await stream_iter  # type: ignore[assignment]

            async for chunk in stream_iter:  # type: ignore[attr-defined]
                if chunk.text:
                    yield chunk.text
        else:
            result = await client.aio.models.generate_content(
                model=self.vertex_model_name,
                contents=contents,
                config=config,
            )
            if result.text:
                yield result.text


class ClaudeOptions(llm.Options):
    """Generation options for Claude models via Vertex AI.

    Uses the official Anthropic SDK parameters.
    """

    temperature: float | None = Field(
        default=None,
        description="Controls randomness. Lower is more deterministic.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens to generate.",
        ge=1,
    )
    top_p: float | None = Field(
        default=None,
        description="Nucleus sampling threshold.",
        ge=0.0,
        le=1.0,
    )
    top_k: int | None = Field(
        default=None,
        description="Top-k sampling parameter.",
        ge=1,
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description="Custom stop sequences.",
    )

    @field_validator("stop_sequences", mode="before")
    @classmethod
    def validate_stop_sequences(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [v]
        return v


@lru_cache(maxsize=1)
def get_anthropic_vertex_client(
    project: str | None = None, region: str | None = None
) -> AnthropicVertex:
    """
    Get a cached AnthropicVertex client.

    Uses Application Default Credentials for authentication.
    Client is cached after first call.
    """
    from anthropic import AnthropicVertex

    return AnthropicVertex(
        project_id=project or get_project_id(),
        region=region or get_location(),
    )


@lru_cache(maxsize=1)
def get_async_anthropic_vertex_client(
    project: str | None = None, region: str | None = None
) -> AsyncAnthropicVertex:
    """
    Get a cached AsyncAnthropicVertex client.

    Uses Application Default Credentials for authentication.
    Client is cached after first call.
    """
    from anthropic import AsyncAnthropicVertex

    return AsyncAnthropicVertex(
        project_id=project or get_project_id(),
        region=region or get_location(),
    )


def build_claude_messages(
    prompt: llm.Prompt, conversation: llm.Conversation | None
) -> list[MessageParam]:
    """Build Anthropic-format messages from prompt and conversation history."""
    messages: list[MessageParam] = []

    if conversation:
        for prev_response in conversation.responses:
            prev = cast(llm.Response, prev_response)
            messages.append({"role": "user", "content": prev.prompt.prompt})
            messages.append({"role": "assistant", "content": prev.text()})

    messages.append({"role": "user", "content": prompt.prompt})

    return messages


async def build_claude_messages_async(
    prompt: llm.Prompt, conversation: llm.AsyncConversation | None
) -> list[MessageParam]:
    """Build Anthropic-format messages from prompt and async conversation history."""
    messages: list[MessageParam] = []

    if conversation:
        for prev_response in conversation.responses:
            prev = cast(llm.AsyncResponse, prev_response)
            messages.append({"role": "user", "content": prev.prompt.prompt})
            messages.append({"role": "assistant", "content": await prev.text()})

    messages.append({"role": "user", "content": prompt.prompt})

    return messages


def _build_claude_kwargs(
    prompt: llm.Prompt, model: str, messages: list[MessageParam]
) -> dict[str, Any]:
    """Build kwargs for Anthropic API call."""
    opts = cast(ClaudeOptions, prompt.options)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": opts.max_tokens,
    }

    if opts.temperature is not None:
        kwargs["temperature"] = opts.temperature
    if opts.top_p is not None:
        kwargs["top_p"] = opts.top_p
    if opts.top_k is not None:
        kwargs["top_k"] = opts.top_k
    if opts.stop_sequences:
        kwargs["stop_sequences"] = opts.stop_sequences
    if prompt.system:
        kwargs["system"] = prompt.system

    return kwargs


class VertexClaudeModel(llm.Model):
    """Synchronous Claude model via Vertex AI using official Anthropic SDK."""

    model_id: str
    vertex_model_name: str
    can_stream: bool = True
    needs_key: str | None = None

    class Options(ClaudeOptions): ...  # type: ignore[override]

    def __init__(self, model_id: str, vertex_model_name: str):
        self.model_id = model_id
        self.vertex_model_name = vertex_model_name

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None = None,
    ) -> Generator[str, None, None]:
        client = get_anthropic_vertex_client()
        messages = build_claude_messages(prompt, conversation)
        kwargs = _build_claude_kwargs(prompt, self.vertex_model_name, messages)

        if stream:
            with client.messages.stream(**kwargs) as stream_response:
                for text in stream_response.text_stream:
                    yield text
        else:
            result = client.messages.create(**kwargs)
            for block in result.content:
                if hasattr(block, "text"):
                    yield block.text  # type: ignore[union-attr]


class AsyncVertexClaudeModel(llm.AsyncModel):
    """Asynchronous Claude model via Vertex AI using official Anthropic SDK."""

    model_id: str
    vertex_model_name: str
    can_stream: bool = True
    needs_key: str | None = None

    class Options(ClaudeOptions): ...  # type: ignore[override]

    def __init__(self, model_id: str, vertex_model_name: str):
        self.model_id = model_id
        self.vertex_model_name = vertex_model_name

    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.AsyncResponse,
        conversation: llm.AsyncConversation | None = None,
    ) -> AsyncGenerator[str, None]:
        client = get_async_anthropic_vertex_client()
        messages = await build_claude_messages_async(prompt, conversation)
        kwargs = _build_claude_kwargs(prompt, self.vertex_model_name, messages)

        if stream:
            async with client.messages.stream(**kwargs) as stream_response:
                async for text in stream_response.text_stream:
                    yield text
        else:
            result = await client.messages.create(**kwargs)
            for block in result.content:
                if hasattr(block, "text"):
                    yield block.text  # type: ignore[union-attr]


# Model definitions
GEMINI_MODELS = [
    # Gemini 3 models are still in preview (rate limits may be low)
    ("gemini-3-pro", "gemini-3-pro-preview"),
    ("gemini-3-flash", "gemini-3-flash-preview"),
    # Gemini 2.5 models
    ("gemini-2.5-pro", "gemini-2.5-pro"),
    ("gemini-2.5-flash", "gemini-2.5-flash"),
    ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite"),
    ("gemini-2.0-flash", "gemini-2.0-flash"),
]

# Claude models use Anthropic SDK format: model-name@version
# See: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude
CLAUDE_MODELS = [
    # Claude 4.5 models (latest)
    ("claude-opus-4.5", "claude-opus-4-5@20251101"),
    ("claude-sonnet-4.5", "claude-sonnet-4-5@20250929"),
    ("claude-haiku-4.5", "claude-haiku-4-5@20251001"),
    # Claude 4.1 models
    ("claude-opus-4.1", "claude-opus-4-1@20250805"),
    # Claude 4 models
    ("claude-sonnet-4", "claude-sonnet-4@20250514"),
    ("claude-opus-4", "claude-opus-4@20250514"),
]


@llm.hookimpl
def register_models(register: Callable[..., object]) -> None:
    """Register all Vertex AI models with the LLM framework."""
    for model_id, vertex_name in GEMINI_MODELS:
        _ = register(
            VertexGeminiModel(model_id, vertex_name),
            AsyncVertexGeminiModel(model_id, vertex_name),
        )

    for model_id, vertex_name in CLAUDE_MODELS:
        _ = register(
            VertexClaudeModel(model_id, vertex_name),
            AsyncVertexClaudeModel(model_id, vertex_name),
        )
