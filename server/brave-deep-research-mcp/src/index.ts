#!/usr/bin/env node
import { startServer } from './server.js';
import { config } from './utils/config.js';

// Display startup message with configuration
console.error(`Starting Brave Deep Research MCP Server
Configuration:
- Puppeteer Headless: ${config.isHeadless}
- Page Timeout: ${config.PAGE_TIMEOUT}ms
- Debug Mode: ${config.isDebugMode}
`);

// Start the server
startServer().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});