# Research Pipeline Testing Guide

This document provides curl commands to manually test the research pipeline components.

## Prerequisites

1. Start the research pipeline server:
```bash
cd /home/siamai/deepsad/local-deep-researcher
python src/main.py
```

2. Ensure ArXiv MCP server is running:
```bash
cd server/arxiv-mcp-server
python -m arxiv_mcp_server
```

3. Start SearXNG (if using web search):
```bash
cd agentic/searxng-docker
docker-compose up -d
```

## Test Cases

### 1. ArXiv Paper Analysis Test

Test the pipeline with an ArXiv paper URL to verify ArXiv detection and paper ID extraction.

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "research_topic": "explain https://arxiv.org/html/2410.21338v2 this paper",
    "max_research_loops": 1
  }'
```

**Expected Behavior:**
- Intent classification: `arxiv_search` 
- Query extraction: `2410.21338` (version stripped)
- MCP server called for paper download and analysis

### 2. ArXiv Paper ID Test

Test with just a paper ID to verify direct ArXiv search.

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "research_topic": "analyze paper 1706.03762 attention mechanism",
    "max_research_loops": 1
  }'
```

**Expected Behavior:**
- Intent classification: `arxiv_search`
- Paper ID detection: `1706.03762`
- Direct ArXiv MCP search

### 3. Web Search Test

Test general web search functionality.

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "research_topic": "latest AI developments 2024 transformer models",
    "max_research_loops": 1
  }'
```

**Expected Behavior:**
- Intent classification: `web_search`
- SearXNG web search execution
- Current content retrieval

### 4. Hybrid Search Test

Test hybrid search combining academic and web sources.

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "research_topic": "transformer architecture performance benchmarks comparison",
    "max_research_loops": 2
  }'
```

**Expected Behavior:**
- Intent classification: `hybrid_search`
- Both ArXiv and web sources used
- Multi-loop research refinement

### 5. Configuration Test

Test different LLM providers and settings.

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "research_topic": "machine learning optimization techniques",
    "max_research_loops": 1,
    "llm_provider": "nvidia_nim",
    "max_papers_per_search": 5
  }'
```

## Health Check Endpoints

### Pipeline Status
```bash
curl -X GET http://localhost:8000/health
```

### Configuration Check
```bash
curl -X GET http://localhost:8000/config
```

## Debugging Commands

### Check Intent Classification Only
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "explain https://arxiv.org/html/2410.21338v2 this paper"
  }'
```

### Test Query Generation Only
```bash
curl -X POST http://localhost:8000/generate-query \
  -H "Content-Type: application/json" \
  -d '{
    "research_topic": "transformer attention mechanisms",
    "search_intent": "arxiv_search"
  }'
```

### ArXiv MCP Server Direct Test
```bash
# Test MCP server connectivity
curl -X POST http://localhost:3001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "2410.21338",
    "max_results": 1
  }'
```

## Expected Response Format

Successful research response should include:

```json
{
  "research_topic": "...",
  "search_intent": "arxiv_search|web_search|hybrid_search",
  "intent_confidence": 0.85,
  "search_query": "optimized search query",
  "search_results": [...],
  "extracted_content": [...],
  "research_summary": "comprehensive analysis...",
  "sources": [...],
  "processing_times": {
    "intent_classification": 0.12,
    "query_generation": 4.85,
    "search_execution": 2.31,
    "content_extraction": 1.45,
    "summarization": 3.21,
    "total": 12.04
  },
  "research_loop_count": 1,
  "research_quality_score": 0.87
}
```

## Error Scenarios to Test

### 1. Invalid ArXiv URL
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "research_topic": "explain https://arxiv.org/invalid/url this paper"
  }'
```

### 2. Network Issues
```bash
# Test with MCP server down
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "research_topic": "explain https://arxiv.org/html/2410.21338v2 this paper"
  }'
```

### 3. Malformed Request
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "invalid_field": "test"
  }'
```

## Performance Testing

### Concurrent Requests
```bash
# Run multiple requests simultaneously
for i in {1..3}; do
  curl -X POST http://localhost:8000/research \
    -H "Content-Type: application/json" \
    -d "{\"research_topic\": \"test query $i\"}" &
done
wait
```

### Large Research Loop
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "research_topic": "comprehensive analysis of transformer architectures",
    "max_research_loops": 5
  }'
```

## Validation Checklist

- [ ] ArXiv URL detection works correctly
- [ ] Paper ID extraction strips version numbers  
- [ ] JSON parsing handles LLM response variations
- [ ] Intent classification routes correctly
- [ ] MCP server integration functional
- [ ] Web search fallback works
- [ ] Error handling graceful
- [ ] Performance within acceptable limits
- [ ] Multi-loop research improves quality
- [ ] All response fields populated

## Troubleshooting

1. **JSON parsing errors**: Check prompt escaping in `prompts.py`
2. **MCP server connection**: Verify server running on correct port
3. **Intent classification fails**: Check model files in `/data/` directory
4. **Async errors**: Verify all async functions properly awaited
5. **Search returns 0 results**: Check ArXiv ID format and MCP compatibility