# llm-gcp-vertex

[llm](https://github.com/simonw/llm) plugin for Google Cloud Vertex AI, providing access to Gemini and Claude models through their official SDKs. If you are looking for support for
Google AI Studio, see the `llm-gemini` plugin.

## Features

- **Gemini models** via the official `google-genai` SDK
- **Claude models** via the official `anthropic[vertex]` SDK  
- Full streaming support for both model families
- Async support for high-throughput applications

## Installation

```bash
llm install llm-gcp-vertex
```

## Prerequisites

### 1. Google Cloud Authentication

This plugin uses Application Default Credentials (ADC):

```bash
gcloud auth application-default login
```

### 2. Configure Project and Location

**Using environment variables:**
```bash
# Falls back to GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION if not set
export LLM_VERTEX_CLOUD_PROJECT="your-project-id"
export LLM_VERTEX_CLOUD_LOCATION="us-central1"  # optional, defaults to us-central1
```

**Or using LLM keys:**
```bash
llm keys set vertex-project
llm keys set vertex-location  # optional
```

## Available Models

### Gemini

| Model ID | Description |
|----------|-------------|
| `gemini-3-pro-preview` | Latest flagship, complex reasoning (preview) |
| `gemini-3-flash-preview` | Latest fast model (preview) |
| `gemini-2.5-pro` | Strong reasoning and coding |
| `gemini-2.5-flash` | Fast and cost-effective |
| `gemini-2.0-flash` | Balanced speed and quality |

### Claude (via Vertex AI)

| Model ID | Description |
|----------|-------------|
| `claude-opus-4.5` | Most capable, complex analysis |
| `claude-sonnet-4.5` | Best for coding and agents |
| `claude-haiku-4.5` | Fast and affordable |
| `claude-opus-4.1` | Extended thinking, agentic tasks |
| `claude-sonnet-4` | Balanced speed and capability |
| `claude-opus-4` | Strong reasoning |

## Usage

### Command Line

```bash
# Gemini
llm -m gemini-2.5-flash "Explain quantum computing"
llm -m gemini-2.5-pro "Write a poem" -o temperature 0.9

# Claude
llm -m claude-sonnet-4.5 "Review this code"
llm -m claude-haiku-4.5 "Summarize this" -o max_tokens 500

# With system prompt
llm -m gemini-2.0-flash "Explain recursion" -s "You are a patient teacher"
```

## Options

### Gemini Options

| Option | Type | Description |
|--------|------|-------------|
| `temperature` | float (0.0-2.0) | Controls randomness |
| `max_output_tokens` | int | Maximum tokens to generate |
| `top_p` | float (0.0-1.0) | Nucleus sampling threshold |
| `top_k` | int | Top-k sampling parameter |

### Claude Options

| Option | Type | Description |
|--------|------|-------------|
| `temperature` | float (0.0-1.0) | Controls randomness |
| `max_tokens` | int | Maximum tokens to generate (default: 4096) |
| `top_p` | float (0.0-1.0) | Nucleus sampling threshold |
| `top_k` | int | Top-k sampling parameter |
| `stop_sequences` | list[str] | Custom stop sequences |

## Development

```bash
git clone https://github.com/ASRagab/llm-gcp-vertex.git
cd llm-gcp-vertex

# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Type check (strict)
uv run basedpyright llm_gcp_vertex.py
```

## Troubleshooting

### "Vertex AI project ID required"

```bash
export LLM_VERTEX_CLOUD_PROJECT="your-project-id"
# or
llm keys set vertex-project
```

### "Could not automatically determine credentials"

```bash
gcloud auth application-default login
```

### Claude models not working

Claude on Vertex AI requires:
1. Enable the Claude API in your GCP project
2. Accept the usage agreement in [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden)
3. Use a [supported region](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude) (e.g., `us-east5`, `europe-west1`)

```bash
export LLM_VERTEX_CLOUD_LOCATION="us-east5"
```

## License

Apache 2.0
