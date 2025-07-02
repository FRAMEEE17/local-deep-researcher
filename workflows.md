# Phase 1: Intent-Driven Initialization (Sequential: →)

START → Intent Classification → Query Generation → Search Strategy

- A1. Research Topic Input → A2. XLM-RoBERTa Intent Classification → A3. Strategy Selection
  - research_pipeline/graph.py:33-82 loads trained model from /data/intent_classifier_xlm.pth
  - Output Mapping:
    - Label 0 (Academic) → "arxiv_search"
    - Label 2 (Hybrid) → "hybrid_search"
    - Others → "web_search"

# Phase 2: ArXiv MCP Server Integration (Sequential → then Concurrent ⇉)

- LangGraph → HTTP Request → MCPO Gateway → MCP Protocol → ArXiv Server → Tools

B1. Pipeline HTTP Call → B2. MCP Server Activation ⇉ B3. Concurrent Tool Execution

  research_pipeline/arxiv_http_client.py
      ↓ HTTP POST to localhost:9937/mcp/call
  MCPO Gateway (OpenWebUI integration)
      ↓ MCP JSON-RPC Protocol
  server/arxiv-mcp-server/src/arxiv_mcp_server/server.py
      ↓ Tool dispatch
  hybrid_search_tool → [ArXiv API ⇉ SearchTheArxiv API ⇉ Jina AI]

  Phase 3: Content Processing & Synthesis (Concurrent ⇉ then Sequential →)

  Multiple APIs ⇉ Content Extraction ⇉ Async Processing → Synthesis → Loop Control

  C1. Concurrent Content Extraction ⇉ C2. Research Synthesis → C3. Iteration Control

  # Concurrent operations (⇉)
  asyncio.gather([
      arxiv_api_search(),           # Via MCP Server
      searchthearxiv_api(),         # Via MCP Server  
      jina_content_extraction()     # Via MCP Server or Direct
  ])

  # Sequential operations (→)
  synthesize_research() → reflect_on_research() → route_next_action()

### 🌊 Data Flow Architecture

- Key Integration Points

  1. MCP Protocol Bridge: research_pipeline/mcp_client.py:169-294
  ArxivMCPClient → JSON-RPC over stdin/stdout → MCP Server
  2. HTTP Gateway: research_pipeline/arxiv_http_client.py
  LangGraph → HTTP POST → localhost:9937 → MCPO → MCP Server
  3. State Management: research_pipeline/state.py:7-48
  ResearchState accumulates: arxiv_results + semantic_results + web_results

- Execution Flow Types

  ### Sequential Operations (→):
  - Intent classification → Query optimization → Strategy selection
  - Research synthesis → Reflection → Next action routing

  ### Concurrent Operations (⇉):
  - Multiple API calls within hybrid search
  - Batch content extraction via asyncio.gather()
  - PDF processing in MCP server

  ### Loop Control Logic

  ┌─> execute_search() → extract_content() → synthesize_research() 
  │                                                    ↓
  └── reflect_on_research() ←── route_research_flow() ←┘
                      ↓
              finalize_research() (EXIT)

### Final Integration Summary

- The two systems work together as a sophisticated hybrid RAG pipeline where:

  1. Intent Classification (XLM-RoBERTa) drives dynamic routing
  2. MCP Protocol enables rich academic content access
  3. Concurrent APIs maximize search coverage and speed
  4. LangGraph Orchestration manages complex multi-step workflows
  5. State Accumulation builds comprehensive research synthesis