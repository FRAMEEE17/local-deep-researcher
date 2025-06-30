#!/usr/bin/env python3
"""
Main test script for research pipeline verification.
Tests the complete search flow: graph.py -> SearchEngines -> execute_arxiv_search_strategy()
"""

import asyncio
import time
import sys
from pathlib import Path
from research_pipeline.configuration import Configuration

default_config = Configuration()
# Add research_pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "research_pipeline"))

async def test_search_flow():
    
    try:
        from research_pipeline.graph import app
        from research_pipeline.state import ResearchStateInput
        from langchain_core.runnables import RunnableConfig
        
        # Test configuration
        config = RunnableConfig(
            configurable={
                "llm_provider": "nvidia_nim",
                "local_llm": "meta/llama-3.1-8b-instruct",
                "search_api": "searxng",
                "max_web_research_loops": 3,
                "fetch_full_page": True,
                "max_papers_per_search": 15,
                "jina_api_key": default_config.jina_api_key
            }
        )
        
        # Test input
        input_data = ResearchStateInput(
            research_topic="could you extract contents in https://arxiv.org/html/2410.21338v2 paper about key experiments and what they're actually solving"
        )
        
        print("Starting pipeline test...")
        start_time = time.time()
        
        # Execute pipeline
        result = await app.ainvoke(input_data, config=config)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Extract results
        final_summary = None
        arxiv_results = []
        web_results = []
        
        # Check what's actually in the LangGraph result
        print(f"DEBUG: result type = {type(result)}")
        print(f"DEBUG: dir(result) = {[attr for attr in dir(result) if not attr.startswith('_')]}")
        
        # Handle LangGraph AddableValuesDict result type
        try:
            # Try to access as dict-like object first
            final_summary = result.get('final_summary') or result.get('running_summary')
            arxiv_results = result.get('arxiv_results', []) or result.get('papers', [])
            web_results = result.get('web_results', [])
            metadata = result.get('research_metadata', {})
            
            # Check all keys in the result
            if hasattr(result, 'keys'):
                available_keys = list(result.keys())
                print(f"DEBUG: Available keys = {available_keys}")
                
                # Check if arxiv_results exists but is empty
                if 'arxiv_results' in available_keys:
                    print(f"DEBUG: arxiv_results = {result['arxiv_results']}")
                
        except (AttributeError, TypeError):
            # Fallback to attribute access for other object types
            final_summary = getattr(result, 'final_summary', None) or getattr(result, 'running_summary', None)
            arxiv_results = getattr(result, 'arxiv_results', [])
            web_results = getattr(result, 'web_results', [])
            metadata = getattr(result, 'research_metadata', {})
        
        # Results
        print(f"Pipeline completed in {duration:.2f}s")
        
        if final_summary:
            print(f"Summary generated: {len(final_summary)} characters")
            print(f"ArXiv papers processed: {len(arxiv_results)}")
            print(f"Web results processed: {len(web_results)}")
            
            # Show research metadata if available
            if metadata:
                source_breakdown = metadata.get('source_breakdown', {})
                print(f"Source breakdown: {source_breakdown}")
            
            # Show summary preview
            print("\nSummary preview:")
            print(final_summary[:300] + "..." if len(final_summary) > 300 else final_summary)
            
            return True
        else:
            print("No summary generated")
            print(f"Result type: {type(result)}")
            if isinstance(result, dict):
                print(f"Available keys: {list(result.keys())}")
            return False
            
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_search_engines_directly():
    """Test SearchEngines class directly."""
    
    try:
        from research_pipeline.search_engines import create_search_engines
        from research_pipeline.configuration import Configuration
        
        print("\nTesting SearchEngines directly...")
        
        config = Configuration()
        search_engines = create_search_engines(config)
        
        result = await search_engines.execute_search_strategy(
            query="paper 2410.21338",
            strategy="arxiv_search",
            max_results=5
        )
        
        if result.get("success"):
            papers = result.get("papers", [])
            print(f"Direct search found {len(papers)} papers")
            if papers:
                print(f"First paper: {papers[0].get('title', 'Unknown')}")
            return True
        else:
            print(f"Direct search failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"Direct search test failed: {e}")
        return False

async def test_arxiv_strategy_directly():
    try:
        from research_pipeline.arxiv_http_client import execute_arxiv_search_strategy
        from research_pipeline.configuration import Configuration
        
        print("\nTesting execute_arxiv_search_strategy directly...")
        
        config = Configuration()
        result = await execute_arxiv_search_strategy(
            query="paper 2410.21338",
            strategy="arxiv_search",
            config=config
        )
        
        if result.get("success"):
            papers = result.get("papers", [])
            print(f"ArXiv strategy found {len(papers)} papers")
            if papers:
                print(f"Paper title: {papers[0].get('title', 'Unknown')}")
            return True
        else:
            print(f"ArXiv strategy failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"ArXiv strategy test failed: {e}")
        return False

async def main():
    """Run all tests."""
    
    print("Research Pipeline Test Suite")
    print("=" * 40)
    
    # Test 1: Full pipeline
    success1 = await test_search_flow()
    
    # Test 2: SearchEngines directly
    success2 = await test_search_engines_directly()
    
    # Test 3: ArXiv strategy directly
    success3 = await test_arxiv_strategy_directly()
    
    # Summary
    print("\nTest Results:")
    print(f"Full pipeline: {'PASS' if success1 else 'FAIL'}")
    print(f"SearchEngines: {'PASS' if success2 else 'FAIL'}")
    print(f"ArXiv strategy: {'PASS' if success3 else 'FAIL'}")
    
    if all([success1, success2, success3]):
        print("\nAll tests passed. Search flow verification complete.")
        return 0
    else:
        print("\nSome tests failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)