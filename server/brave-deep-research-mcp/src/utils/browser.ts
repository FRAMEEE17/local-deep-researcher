import puppeteer, { Browser } from 'puppeteer';
import { config, debug } from './config.js';

// Singleton browser instance
let browserInstance: Browser | null = null;

/**
 * Get a browser instance, creating one if it doesn't exist
 */
export async function getBrowser(): Promise<Browser> {
  if (!browserInstance) {
    debug('Launching new browser instance');
    
    const options = {
      headless: config.isHeadless,
      args: [
        '--disable-setuid-sandbox',
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--no-first-run',
        '--no-zygote',
        '--disable-gpu'
      ]
    };
    
    browserInstance = await puppeteer.launch(options);
    
    // Handle browser closing
    browserInstance.on('disconnected', () => {
      debug('Browser disconnected');
      browserInstance = null;
    });
  }
  
  return browserInstance;
}

/**
 * Close the browser instance if it exists
 */
export async function closeBrowser(): Promise<void> {
  if (browserInstance) {
    debug('Closing browser instance');
    await browserInstance.close();
    browserInstance = null;
  }
}

// Handle process exit to close browser gracefully
process.on('SIGINT', async () => {
  await closeBrowser();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await closeBrowser();
  process.exit(0);
});