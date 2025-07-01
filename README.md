  # Quick Start Commands

  ### Option 1: Automated Startup
  ```
  cd /home/siamai/deepsad/local-deep-researcher
  ./start_research_stack.sh
  ```
  ### Option 2: Manual Startup

  #### Terminal 1: Start MCPO (if not running)
  ```
  mcpo --port 9937 --config server/arxiv-mcp-server/config.json
  ```

  #### Terminal 2: Start Pipelines Server  
  ```
  python start_pipelines.py
  ```

  #### Terminal 3: Check services
  ```
  curl http://localhost:9937/docs  # MCPO endpoints
  curl http://localhost:9097      # Pipelines server
  ```

  ### 🔗 OpenWebUI Configuration

- 1. Settings → Connections → OpenAI API
- 2. API Base URL: http://localhost:9097
- 3. API Key: 0p3n-w3bu!
- 4. Save & Test Connection

  ###  Available Functions

- OUR pipeline exposes these tools via function calling:

  - search_arxiv_papers(query, max_results, search_method) - Hybrid search
  - download_arxiv_paper(arxiv_id) - Download & convert to markdown
  - read_arxiv_paper(arxiv_id) - Read downloaded content
  - list_downloaded_papers() - List all downloaded papers

  ### 🎮 Example Usage

- In OpenWebUI chat:
  - "Search for papers about transformer neural networks"
  - "Download paper 1706.03762"
  - "Read the attention is all you need paper"