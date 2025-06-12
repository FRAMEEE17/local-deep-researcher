# @suthio/brave-deep-research-mcp

A Model Context Protocol (MCP) server that combines Brave Search with Puppeteer-powered content extraction for deep research capabilities. This server allows AI assistants to perform comprehensive web searches by not only retrieving search results but also visiting the pages to extract full content and explore linked pages.

## Comparison with Standard Brave Search MCP Server

### Standard Brave Search MCP Server:
- **Search Capability**: Uses the Brave Search API to perform basic web searches
- **Data Retrieval**: Returns only the search results (title, URL, and snippet) provided by the API
- **Content Depth**: No access to full webpage content beyond the search snippets
- **Page Exploration**: No ability to visit pages or follow links
- **Information Scope**: Limited to the brief information available in search results
- **Content Processing**: No content extraction or cleaning capabilities
- **Customization**: Limited to basic search parameters (query, count, offset)
- **Use Case**: Best for quick searches where only an overview is needed

### Brave Deep Research MCP Server (this project):
- **Search Capability**: Uses Brave Search API for initial results, then enhances with web scraping
- **Data Retrieval**: Extracts complete page content from each search result
- **Content Depth**: Provides full webpage content with main text extraction
- **Page Exploration**: Can traverse links to explore related content at configurable depths
- **Information Scope**: Accesses comprehensive information across multiple related pages
- **Content Processing**: Intelligently identifies and extracts main content, filtering out navigation, ads, footers, etc.
- **Customization**: Configurable depth of exploration, result count, headless mode, and timeouts
- **Use Case**: Ideal for in-depth research requiring detailed information and context

### Practical Differences in an Example Query

For a query like "climate change mitigation technologies":

**Standard Brave Search MCP**:
```
Title: "Latest Climate Change Mitigation Technologies - Example Site"
URL: "https://example.com/climate-tech"
Snippet: "Various technologies are being developed to mitigate climate change, including carbon capture..."
```
(Limited to just these search result snippets)

**Brave Deep Research MCP**:
```
# Latest Climate Change Mitigation Technologies - Example Site
URL: https://example.com/climate-tech

## Content
Carbon capture and storage (CCS) technology has advanced significantly in recent years. The latest direct air capture facilities can now remove CO2 at a cost of $250 per ton, down from $600 just five years ago. Implementation challenges remain, including...

[Followed by several pages of detailed content from the original page and linked pages]
```

## Features

- **Deep Search**: Go beyond search results to extract complete page content
- **Configurable Depth**: Specify how many levels of links to follow from initial results
- **Content Extraction**: Intelligently identify and extract main content from pages
- **Metadata Extraction**: Get titles, descriptions, and structured content
- **Debug Mode**: Configurable logging for troubleshooting
- **Headless Mode Toggle**: Run browser in visible or headless mode

## Installation

```bash
# Install from npm
npm install -g @suthio/brave-deep-research-mcp

# Or clone the repository
git clone https://github.com/suthio/brave-deep-research-mcp.git
cd brave-deep-research-mcp
npm install
npm run build
```

## Configuration

Create a `.env` file based on the provided `.env.example`:

```bash
# Copy the example env file
cp .env.example .env

# Edit the file to add your Brave API key and other settings
nano .env
```

### Environment Variables

- `BRAVE_API_KEY`: Your Brave Search API key (required)
- `PUPPETEER_HEADLESS`: Whether to run Puppeteer in headless mode (default: true)
- `PAGE_TIMEOUT`: Timeout for page loading in milliseconds (default: 30000)
- `DEBUG_MODE`: Enable detailed debug logging (default: false)

## Usage

### Running from command line

```bash
# If installed globally via npm
brave-deep-research-mcp

# Or run directly from the package
npx @suthio/brave-deep-research-mcp

# Or run locally after cloning
npm start
```

### Using with Claude for Desktop

To use this server with Claude for Desktop:

1. Install the package:
```bash
npm install -g @suthio/brave-deep-research-mcp
```

2. Edit the Claude for Desktop configuration file:
   - On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - On Windows: `%APPDATA%\Claude\claude_desktop_config.json`

3. Add the following to the `mcpServers` section:
```json
{
  "mcpServers": {
    "brave-deep-research": {
      "command": "npx",
      "args": ["@suthio/brave-deep-research-mcp"],
      "env": {
        "BRAVE_API_KEY": "your_brave_api_key_here",
        "PUPPETEER_HEADLESS": "true"
      }
    }
  }
}
```

4. Restart Claude for Desktop
5. You can now use the deep-search tool in your conversations

### Example Queries

- "Use deep-search to research the latest developments in quantum computing"
- "Perform a deep search on climate change mitigation strategies with depth 2"
- "Deep search for information about sustainable architecture, with 5 results"

## Tool Parameters

The `deep-search` tool accepts the following parameters:

- `query` (required): The search query
- `results` (optional): Number of search results to process (default: 3, max: 10)
- `depth` (optional): Depth of link traversal for each result (default: 1, max: 3)

## Development

```bash
# Clone the repository
git clone https://github.com/suthio/brave-deep-research-mcp.git
cd brave-deep-research-mcp

# Install dependencies
npm install

# Run in development mode
npm run dev

# Build the project
npm run build
```

## How It Works

1. The tool first performs a search using the Brave Search API to get initial results
2. For each search result, it launches a Puppeteer browser to visit the page
3. It extracts the main content, metadata, and links from each page
4. If depth > 1, it follows links on the page and repeats the process
5. All extracted content is formatted and returned to the AI assistant

## License

MIT