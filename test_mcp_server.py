#!/usr/bin/env python3
import asyncio
import sys
import os

# Add research pipeline to path
sys.path.insert(0, '/home/siamai/deepsad/local-deep-researcher/src')

from research_pipeline.mcp_client import ArxivMCPClient


async def test_mcp_server():   
    print("...Testing ArXiv MCP Server")
    print("=" * 50)
    
    # Create client
    client = ArxivMCPClient()
    
    try:
        # Start the server
        print("...Starting MCP server...")
        success = await client.start_server()
        
        if not success:
            print("❌ Failed to start MCP server")
            return
        
        print("✅ MCP server started successfully")
        
        # Test 1: List available tools
        print("\n...Testing: List available tools")
        tools = await client.list_tools()
        print(f"Available tools: {[tool.get('name', 'unknown') for tool in tools]}")
        
        # Test 2: Basic ArXiv search
        print("\n...Testing: Basic ArXiv search")
        search_result = await client.search_papers(
            query="large language models",
            max_results=3
        )
        
        if search_result:
            print(f"✅ Found {len(search_result)} papers")
            for i, paper in enumerate(search_result[:2], 1):
                print(f"  {i}. {paper.get('title', 'No title')[:60]}...")
        else:
            print("❌ No papers found")
        
        # Test 3: Test hybrid search 
        print("\n...Testing: Hybrid search tool")
        try:
            hybrid_result = await client.call_tool("hybrid_search", {
                "query": "transformer architecture",
                "max_results": 3,
                "search_method": "hybrid",
                "include_content": False
            })
            
            print(f"✅ Hybrid search result: {hybrid_result.get('content', 'No content')[:100]}...")
            
        except Exception as e:
            print(f"⚠️  Hybrid search test failed: {e}")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        # Cleanup
        print("\n...Stopping MCP server...")
        await client.stop_server()
        print("✅ Server stopped")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())