#!/usr/bin/env python3
"""
Quick Test Script for Research Pipeline
Debug why queries might get stuck
"""

import sys
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_paper_query():
    """Test the specific paper query that got stuck."""
    
    print("🔍 Testing ArXiv paper query: 2410.21338v2")
    print("=" * 50)
    
    try:
        # Test 1: Check if MCPO is running
        print("1. Testing MCPO connection...")
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get("http://localhost:9937/docs", timeout=5) as response:
                    print(f"   ✅ MCPO is running (status: {response.status})")
            except Exception as e:
                print(f"   ❌ MCPO connection failed: {e}")
                return
        
        # Test 2: Direct ArXiv search
        print("\n2. Testing direct ArXiv search...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:9937/arxiv-mcp-server/search_papers",
                    json={"query": "2410.21338", "max_results": 3},
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        papers = data.get('papers', [])
                        print(f"   ✅ Found {len(papers)} papers")
                        for paper in papers[:2]:
                            print(f"      - {paper.get('title', 'Unknown')[:60]}...")
                    else:
                        print(f"   ❌ Search failed (status: {response.status})")
        except Exception as e:
            print(f"   ❌ ArXiv search failed: {e}")
        
        # Test 3: Research pipeline components
        print("\n3. Testing pipeline components...")
        
        # Check intent classifier
        try:
            from research_pipeline.intent_classifier import classify_query_intent
            from research_pipeline.configuration import Configuration
            
            config = Configuration()
            result = classify_query_intent("explain arxiv paper 2410.21338v2", config)
            intent = result.get("routing_strategy", "unknown")
            confidence = result.get("confidence", 0.0)
            print(f"   ✅ Intent: {intent} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"   ❌ Intent classifier failed: {e}")
        
        # Test 4: Full pipeline with timeout
        print("\n4. Testing full pipeline (with timeout)...")
        try:
            from research_pipeline.graph import app
            from research_pipeline.state import ResearchStateInput
            from langchain_core.runnables import RunnableConfig
            
            input_data = ResearchStateInput(
                research_topic="explain https://arxiv.org/html/2410.21338v2 this paper"
            )
            
            config = RunnableConfig(
                configurable={
                    "max_web_research_loops": 1,  # Limit iterations
                    "local_llm": "deepseek-r1:70b",
                    "llm_provider": "ollama",
                }
            )
            
            print("   🔄 Running pipeline...")
            start_time = time.time()
            
            # Add timeout to prevent hanging
            try:
                result = await asyncio.wait_for(
                    app.ainvoke(input_data, config=config),
                    timeout=60  # 60 second timeout
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                print(f"   ✅ Pipeline completed in {duration:.2f}s")
                
                # Check results
                if hasattr(result, 'final_summary') and result.final_summary:
                    summary_length = len(result.final_summary)
                    print(f"   📄 Generated summary: {summary_length} characters")
                    print(f"   📊 ArXiv results: {len(getattr(result, 'arxiv_results', []))}")
                    print(f"   🌐 Web results: {len(getattr(result, 'web_results', []))}")
                else:
                    print("   ⚠️ No summary generated")
                    
            except asyncio.TimeoutError:
                print("   ❌ Pipeline timed out after 60 seconds")
                print("   💡 This suggests the pipeline is hanging somewhere")
                
        except Exception as e:
            print(f"   ❌ Pipeline test failed: {e}")
        
        # Test 5: Check LLM connections
        print("\n5. Testing LLM connections...")
        
        # Test Ollama
        # try:
        #     from langchain_ollama import ChatOllama
            
        #     llm = ChatOllama(
        #         model="deepseek-r1:70b",
        #         base_url="http://localhost:11434",
        #         temperature=0.7
        #     )
            
        #     response = await llm.ainvoke("Hello, this is a test. Respond with 'Ollama working'.")
        #     print(f"   ✅ Ollama: {response.content[:50]}...")
            
        # except Exception as e:
        #     print(f"   ❌ Ollama test failed: {e}")
        
        # Test NVIDIA NIM (if API key available)
        try:
            import os
            if os.getenv("NVIDIA_API_KEY"):
                from research_pipeline.nvidia_nim import ChatNVIDIANIM
                
                nvidia_llm = ChatNVIDIANIM(
                    model="qwen/qwen3-235b-a22b",
                    temperature=0.2
                )
                
                response = await nvidia_llm.ainvoke("Hello, respond with 'NVIDIA NIM working'.")
                print(f"   ✅ NVIDIA NIM: {response.content[:50]}...")
            else:
                print("   ⚠️ NVIDIA NIM: No API key (set NVIDIA_API_KEY)")
                
        except Exception as e:
            print(f"   ❌ NVIDIA NIM test failed: {e}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n🔍 Debugging Tips:")
    print("1. Check if MCPO is running: mcpo --port 9937 --config server/arxiv-mcp-server/config.json")
    print("2. Check if Ollama is running: ollama serve")
    print("3. Check model availability: ollama list")
    print("4. Check logs for specific error messages")

if __name__ == "__main__":
    asyncio.run(test_paper_query())