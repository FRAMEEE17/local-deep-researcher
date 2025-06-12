# Setting Up With Claude for Desktop

This guide will help you configure Claude for Desktop to use the @suthio/brave-deep-research-mcp Server.

## Prerequisites

1. Make sure you have [Node.js](https://nodejs.org/) installed (version 16 or higher)
2. Get a Brave Search API key from [Brave Search API](https://brave.com/search/api/)

## Installation Options

### Option 1: Install from npm (Recommended)

Install the package globally:
```bash
npm install -g @suthio/brave-deep-research-mcp
```

### Option 2: Clone and Build

If you prefer to build from source:
```bash
git clone https://github.com/suthio/brave-deep-research-mcp.git
cd brave-deep-research-mcp
npm install
npm run build
```

## Configuration Steps

1. Create or edit the Claude for Desktop configuration file:
   - On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - On Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the Brave Deep Research MCP Server to the `mcpServers` section:

### If installed from npm:

```json
{
  "mcpServers": {
    "brave-deep-research": {
      "command": "npx",
      "args": ["@suthio/brave-deep-research-mcp"],
      "env": {
        "BRAVE_API_KEY": "your_brave_api_key_here",
        "PUPPETEER_HEADLESS": "true",
        "PAGE_TIMEOUT": "30000",
        "DEBUG_MODE": "false"
      }
    }
  }
}
```

### If built from source:

```json
{
  "mcpServers": {
    "brave-deep-research": {
      "command": "node",
      "args": ["/absolute/path/to/brave-deep-research-mcp/build/index.js"],
      "env": {
        "BRAVE_API_KEY": "your_brave_api_key_here",
        "PUPPETEER_HEADLESS": "true",
        "PAGE_TIMEOUT": "30000",
        "DEBUG_MODE": "false"
      }
    }
  }
}
```

3. Replace `your_brave_api_key_here` with your actual Brave Search API key.

4. Restart Claude for Desktop.

## Verifying the Configuration

1. After restarting Claude for Desktop, look for the tools icon (hammer) in the chat interface.
2. Click on the tools icon to see the available tools.
3. You should see the "deep-search" tool listed.

## Using the Deep Search Tool

You can now ask Claude to use the deep search tool by saying something like:

- "Can you use deep-search to research quantum computing advancements in 2024?"
- "Please perform a deep search on climate change mitigation with depth 2"
- "I need detailed information about sustainable architecture. Use deep-search with 5 results"

## Troubleshooting

If you encounter issues:

1. Check the Claude logs:
   - macOS: `~/Library/Logs/Claude/mcp-server-brave-deep-research.log`
   - Windows: `%APPDATA%\Claude\logs\mcp-server-brave-deep-research.log`

2. Make sure your Brave API key is valid.

3. Try setting `DEBUG_MODE` to `true` to get more detailed logs.

4. If you want to see the browser in action, set `PUPPETEER_HEADLESS` to `false`.