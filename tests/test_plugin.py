"""Tests for llm-gcp-vertex plugin."""

import os
from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

import pytest

import llm
import llm_gcp_vertex


@dataclass
class _Options:
    temperature: float | None
    max_output_tokens: int | None
    top_p: float | None
    top_k: int | None


@dataclass
class _PromptLike:
    options: _Options
    system: str | None


class TestConfiguration:
    """Tests for configuration functions."""

    def test_get_project_id_from_env(self):
        """Project ID should be read from environment variable."""
        with patch.dict(os.environ, {"LLM_VERTEX_CLOUD_PROJECT": "test-project"}):
            with patch("llm.get_key", return_value=None):
                assert llm_gcp_vertex.get_project_id() == "test-project"

    def test_get_project_id_from_legacy_env(self):
        """Project ID should fall back to legacy GOOGLE_CLOUD_PROJECT."""
        with patch.dict(
            os.environ, {"GOOGLE_CLOUD_PROJECT": "legacy-project"}, clear=True
        ):
            with patch("llm.get_key", return_value=None):
                assert llm_gcp_vertex.get_project_id() == "legacy-project"

    def test_get_project_id_from_llm_keys(self):
        """Project ID from LLM keys should take priority over env var."""
        with patch.dict(os.environ, {"LLM_VERTEX_CLOUD_PROJECT": "env-project"}):
            with patch("llm.get_key", return_value="llm-project"):
                assert llm_gcp_vertex.get_project_id() == "llm-project"

    def test_get_project_id_missing_raises(self):
        """Missing project ID should raise ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("llm.get_key", return_value=None):
                with pytest.raises(ValueError, match="project ID required"):
                    _ = llm_gcp_vertex.get_project_id()

    def test_get_location_default(self):
        """Location should default to us-central1."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("llm.get_key", return_value=None):
                assert llm_gcp_vertex.get_location() == "us-central1"

    def test_get_location_from_env(self):
        """Location should be read from environment variable."""
        with patch.dict(os.environ, {"LLM_VERTEX_CLOUD_LOCATION": "europe-west1"}):
            with patch("llm.get_key", return_value=None):
                assert llm_gcp_vertex.get_location() == "europe-west1"

    def test_get_location_from_legacy_env(self):
        """Location should fall back to legacy GOOGLE_CLOUD_LOCATION."""
        with patch.dict(
            os.environ, {"GOOGLE_CLOUD_LOCATION": "legacy-location"}, clear=True
        ):
            with patch("llm.get_key", return_value=None):
                assert llm_gcp_vertex.get_location() == "legacy-location"

    def test_get_location_from_llm_keys(self):
        """Location from LLM keys should take priority."""
        with patch.dict(os.environ, {"LLM_VERTEX_CLOUD_LOCATION": "env-location"}):
            with patch("llm.get_key", return_value="llm-location"):
                assert llm_gcp_vertex.get_location() == "llm-location"


class TestModelDefinitions:
    """Tests for model definitions."""

    def test_gemini_models_defined(self):
        """Gemini models should be defined."""
        assert len(llm_gcp_vertex.GEMINI_MODELS) > 0
        model_ids = [m[0] for m in llm_gcp_vertex.GEMINI_MODELS]
        assert "gemini-2.0-flash" in model_ids
        assert "gemini-3-pro" in model_ids
        assert "gemini-3-flash" in model_ids

    def test_claude_models_defined(self):
        """Claude models should be defined."""
        assert len(llm_gcp_vertex.CLAUDE_MODELS) > 0
        model_ids = [m[0] for m in llm_gcp_vertex.CLAUDE_MODELS]
        # Claude 4.5 models (latest)
        assert "claude-opus-4.5" in model_ids
        assert "claude-sonnet-4.5" in model_ids
        assert "claude-haiku-4.5" in model_ids
        # Claude 4 models
        assert "claude-sonnet-4" in model_ids
        assert "claude-opus-4" in model_ids

    def test_model_id_format(self):
        """All model IDs should use clean naming without prefix."""
        for model_id, _ in llm_gcp_vertex.GEMINI_MODELS:
            assert model_id.startswith("gemini-"), (
                f"{model_id} should start with gemini-"
            )

        for model_id, _ in llm_gcp_vertex.CLAUDE_MODELS:
            assert model_id.startswith("claude-"), (
                f"{model_id} should start with claude-"
            )


class TestVertexGeminiModel:
    """Tests for VertexGeminiModel class."""

    def test_model_attributes(self):
        """Model should have required attributes."""
        model = llm_gcp_vertex.VertexGeminiModel(
            "vertex-test",
            "test-model",
        )
        assert model.model_id == "vertex-test"
        assert model.vertex_model_name == "test-model"
        assert model.can_stream is True

    def test_build_config_empty(self):
        """Empty options should return None config."""
        prompt = _PromptLike(
            options=_Options(
                temperature=None,
                max_output_tokens=None,
                top_p=None,
                top_k=None,
            ),
            system=None,
        )

        config = llm_gcp_vertex.build_config(cast(llm.Prompt, cast(object, prompt)))
        assert config is None

    def test_build_config_with_options(self):
        """Config should include provided options."""
        prompt = _PromptLike(
            options=_Options(
                temperature=0.7,
                max_output_tokens=1000,
                top_p=0.9,
                top_k=40,
            ),
            system="You are helpful",
        )

        config = llm_gcp_vertex.build_config(cast(llm.Prompt, cast(object, prompt)))
        assert config is not None

        assert config.temperature == 0.7
        assert config.max_output_tokens == 1000
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert cast(str, config.system_instruction) == "You are helpful"


class TestVertexClaudeModel:
    """Tests for VertexClaudeModel class."""

    def test_model_attributes(self):
        """Model should have required attributes."""
        model = llm_gcp_vertex.VertexClaudeModel(
            "vertex-claude-test",
            "claude-test",
        )
        assert model.model_id == "vertex-claude-test"
        assert model.vertex_model_name == "claude-test"
        assert model.can_stream is True


class TestAsyncModels:
    """Tests for async model classes."""

    def test_async_gemini_model_attributes(self):
        """Async Gemini model should have required attributes."""
        model = llm_gcp_vertex.AsyncVertexGeminiModel(
            "vertex-test",
            "test-model",
        )
        assert model.model_id == "vertex-test"
        assert model.can_stream is True

    def test_async_claude_model_attributes(self):
        """Async Claude model should have required attributes."""
        model = llm_gcp_vertex.AsyncVertexClaudeModel(
            "vertex-claude-test",
            "claude-test",
        )
        assert model.model_id == "vertex-claude-test"
        assert model.can_stream is True


class TestModelRegistration:
    """Tests for model registration."""

    def test_register_models_called(self):
        """register_models should register all models."""
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def register(*args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))

        llm_gcp_vertex.register_models(register)

        expected_calls = len(llm_gcp_vertex.GEMINI_MODELS) + len(
            llm_gcp_vertex.CLAUDE_MODELS
        )
        assert len(calls) == expected_calls
