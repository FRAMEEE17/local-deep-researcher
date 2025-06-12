import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { 
  CallToolRequestSchema,
  ListToolsRequestSchema
} from '@modelcontextprotocol/sdk/types.js';
import { deepSearchTool } from './tools/deep-search.js';
import { debug } from './utils/config.js';
import { closeBrowser } from './utils/browser.js';

/**
 * Create and initialize the MCP server
 */
export async function createServer() {
  debug('Creating MCP server');
  
  // Create server instance
  const server = new Server({
    name: "brave-deep-research-mcp",
    version: "1.0.0"
  }, {
    capabilities: {
      tools: {}
    }
  });
  
  // Register the tool for listing
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: [
        {
          name: deepSearchTool.name,
          description: deepSearchTool.description,
          inputSchema: deepSearchTool.inputSchema
        }
      ]
    };
  });

  // Register the deep-search tool handler
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    // Only handle our tool
    if (request.params.name !== deepSearchTool.name) {
      return {
        isError: true,
        content: [
          {
            type: "text",
            text: `Unknown tool: ${request.params.name}`
          }
        ]
      };
    }
    
    // Execute the tool
    const result = await deepSearchTool.handler(request.params.arguments || {});
    return result;
  });
  
  debug('Registered deep-search tool');
  
  // Set up shutdown handler
  const shutdown = async () => {
    debug('Shutting down server');
    await closeBrowser();
    process.exit(0);
  };
  
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
  
  return server;
}

/**
 * Start the server using stdio transport
 */
export async function startServer() {
  try {
    const server = await createServer();
    
    // Create and connect transport
    const transport = new StdioServerTransport();
    await server.connect(transport);
    
    debug('MCP server running with stdio transport');
  } catch (error) {
    console.error('Error starting server:', error);
    await closeBrowser();
    process.exit(1);
  }
}