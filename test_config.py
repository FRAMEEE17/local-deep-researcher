#!/usr/bin/env python3
"""
Test configuration parsing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config():
    """Test configuration creation."""
    
    try:
        from research_pipeline.configuration import Configuration
        from langchain_core.runnables import RunnableConfig
        
        print("Testing basic configuration...")
        config = Configuration()
        print(f"✅ Basic config created")
        print(f"   search_api: {config.search_api}")
        print(f"   llm_provider: {config.llm_provider}")
        
        print("\nTesting from_runnable_config...")
        runnable_config = RunnableConfig(
            configurable={
                "llm_provider": "nvidia_nim",
                "local_llm": "qwen/qwen3-235b-a22b",
                "search_api": "searxng"
            }
        )
        
        config2 = Configuration.from_runnable_config(runnable_config)
        print(f"✅ Runnable config created")
        print(f"   search_api: {config2.search_api}")
        print(f"   llm_provider: {config2.llm_provider}")
        print(f"   local_llm: {config2.local_llm}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config()