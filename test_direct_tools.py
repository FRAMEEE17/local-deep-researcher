#!/usr/bin/env python3
import sys
import asyncio

# Add the server path
sys.path.insert(0, '/home/siamai/deepsad/local-deep-researcher/server/arxiv-mcp-server/src')

from arxiv_mcp_server.tools import search_tool, hybrid_search_tool, handle_search, handle_hybrid_search


async def test_tools_directly():
    # Test 1: Check tool definitions
    print(f"Search tool: {search_tool.name} - {search_tool.description}")
    print(f"Hybrid tool: {hybrid_search_tool.name} - {hybrid_search_tool.description}")
    
    # Test 2: Call basic search directly
    print("\nTesting basic search handler...")
    try:
        result = await handle_search({
            "query": "machine learning",
            "max_results": 2
        })
        
        print(f"✅ Search result type: {type(result)}")
        if result and len(result) > 0:
            # Parse the TextContent
            content = result[0].text
            print(f"✅ Content preview: {content[:100]}...")
        else:
            print("❌ No results returned")
    
    except Exception as e:
        print(f"❌ Search error: {e}")
    
    # Test 3: Test hybrid search directly
    print("\nTesting hybrid search handler...")
    try:
        result = await handle_hybrid_search({
            "query": "transformer neural networks",
            "max_results": 2,
            "search_method": "arxiv_only",
            "include_content": False
        })
        
        print(f"✅ Hybrid search result type: {type(result)}")
        if result and len(result) > 0:
            content = result[0].text
            print(f"✅ Content preview: {content[:100]}...")
        else:
            print("❌ No results returned")
    
    except Exception as e:
        print(f"❌ Hybrid search error: {e}")
    
    print("\n✅ Direct tools test completed!")


if __name__ == "__main__":
    asyncio.run(test_tools_directly())