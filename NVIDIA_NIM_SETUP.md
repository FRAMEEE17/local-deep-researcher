# NVIDIA NIM Integration Setup

## Overview

NVIDIA NIM (NVIDIA Inference Microservices) support has been added to the research pipeline alongside Ollama and LMStudio.

## Configuration

### 1. Environment Variables

Set your NVIDIA API key:
```bash
export NVIDIA_API_KEY="your-nvidia-api-key-here"
```

### 2. Available Models

Popular NVIDIA NIM models:
- `qwen/qwen3-235b-a22b` (recommended for reasoning)
- `meta/llama-3.1-405b-instruct`
- `meta/llama-3.1-70b-instruct`
- `meta/llama-3.1-8b-instruct` (faster)
- `nvidia/llama-3.1-nemotron-70b-instruct`

### 3. Pipeline Configuration

Use NVIDIA NIM in your research pipeline:

```python
from research_pipeline.graph import app
from research_pipeline.state import ResearchStateInput
from langchain_core.runnables import RunnableConfig

# Configure pipeline to use NVIDIA NIM
config = RunnableConfig(
    configurable={
        "llm_provider": "nvidia_nim",
        "local_llm": "qwen/qwen3-235b-a22b",
        "nvidia_api_key": None,  # Uses NVIDIA_API_KEY env var
        "max_web_research_loops": 2,
    }
)

# Run pipeline
input_data = ResearchStateInput(
    research_topic="your research topic"
)

result = await app.ainvoke(input_data, config=config)
```

### 4. Direct Usage

Use NVIDIA NIM directly:

```python
from research_pipeline.nvidia_nim import ChatNVIDIANIM

# With reasoning (recommended for research)
llm = ChatNVIDIANIM(
    model="qwen/qwen3-235b-a22b",
    temperature=0.2,
    enable_reasoning=True
)

# Fast model for quick responses
llm_fast = ChatNVIDIANIM(
    model="meta/llama-3.1-8b-instruct", 
    temperature=0.1,
    enable_reasoning=False
)
```

## Features

### Reasoning Support
- **Enabled by default** for research synthesis
- **Disabled for JSON output** (query generation, reflection)
- Uses `chat_template_kwargs: {"thinking": True}`

### Temperature Settings
- **Query Generation**: 0.3 (deterministic)
- **Research Synthesis**: 0.1 (factual)
- **Reflection**: 0.4 (creative)

### Automatic Fallbacks
- Falls back to Ollama if NVIDIA NIM fails
- Graceful error handling for API issues

## Testing

Run the test scripts:

```bash
# Quick test
python quick_test.py

# Full pipeline test  
python test_research_pipeline.py

# NVIDIA NIM specific test
python nvidia_nim_example.py
```

## Provider Comparison

| Provider | Strengths | Use Cases |
|----------|-----------|-----------|
| **Ollama** | Local, fast, no API costs | Development, privacy |
| **LMStudio** | Local, OpenAI compatible | Local deployment |
| **NVIDIA NIM** | Cloud, reasoning, latest models | Production, research |

## Configuration Options

All NVIDIA NIM settings in `configuration.py`:

```python
llm_provider: "nvidia_nim"
nvidia_nim_base_url: "https://integrate.api.nvidia.com/v1"
nvidia_api_key: Optional[str]  # Or use NVIDIA_API_KEY env var
local_llm: "qwen/qwen3-235b-a22b"  # Model name
```

## Troubleshooting

1. **API Key Issues**:
   ```bash
   echo $NVIDIA_API_KEY  # Check if set
   ```

2. **Model Not Found**:
   - Verify model name spelling
   - Check NVIDIA NIM model availability

3. **Rate Limits**:
   - NVIDIA NIM has usage limits
   - Consider using faster models for development

4. **Fallback to Ollama**:
   - Pipeline automatically falls back if NVIDIA NIM fails
   - Check logs for specific error messages