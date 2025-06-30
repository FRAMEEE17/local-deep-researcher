"""
NVIDIA NIM API Integration for Research Pipeline
"""

import json
import logging
import os
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field

logger = logging.getLogger(__name__)

class ChatNVIDIANIM(ChatOpenAI):
    """Chat model that uses NVIDIA NIM API with reasoning support."""
    
    enable_reasoning: bool = Field(default=True, description="Enable reasoning/thinking output")
    
    def __init__(
        self,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "meta/llama-3.1-8b-instruct",
        temperature: float = 0.5,
        top_p: float = 0.7,
        # max_tokens: int = 8192,
        timeout: float = 60.0,  # Add 60 second timeout
        nvidia_api_key: Optional[str] = None,
        enable_reasoning: bool = False,
        **kwargs: Any,
    ):
        """Initialize NVIDIA NIM client."""
        
        # Get API key from parameter or environment
        api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY", "")
        
        # Initialize the base class
        super().__init__(
            base_url=base_url,
            model=model,
            temperature=temperature,
            top_p=top_p,
            # max_tokens=max_tokens,
            timeout=timeout,  # Pass timeout to parent
            api_key=api_key,
            **kwargs,
        )
        
        self.enable_reasoning = enable_reasoning
        logger.info(f"Initialized NVIDIA NIM client with model: {model}")
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using NVIDIA NIM API."""
        
        # Add reasoning support if enabled
        if self.enable_reasoning:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"thinking": True},
                **kwargs.get("extra_body", {})
            }
        
        return super()._generate(messages, stop, run_manager, **kwargs)