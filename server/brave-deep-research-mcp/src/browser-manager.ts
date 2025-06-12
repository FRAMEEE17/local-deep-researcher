import puppeteer, { Browser, Page } from 'puppeteer';
import { debug } from './utils/config.js';

/**
 * Singleton browser manager to handle Puppeteer instances
 */
class BrowserManager {
  private static instance: BrowserManager;
  private browser: Browser | null = null;
  private isInitializing = false;
  private initPromise: Promise<Browser> | null = null;

  private constructor() {}

  /**
   * Get the singleton instance of BrowserManager
   */
  public static getInstance(): BrowserManager {
    if (!BrowserManager.instance) {
      BrowserManager.instance = new BrowserManager();
    }
    return BrowserManager.instance;
  }

  /**
   * Initialize the browser if not already initialized
   */
  public async getBrowser(): Promise<Browser> {
    if (this.browser) {
      return this.browser;
    }

    if (this.isInitializing) {
      // If initialization is in progress, return the promise
      return this.initPromise!;
    }

    this.isInitializing = true;
    this.initPromise = this.initBrowser();

    try {
      this.browser = await this.initPromise;
      return this.browser;
    } finally {
      this.isInitializing = false;
      this.initPromise = null;
    }
  }

  /**
   * Initialize a new browser instance
   */
  private async initBrowser(): Promise<Browser> {
    const headless = process.env.PUPPETEER_HEADLESS !== 'false';
    debug(`Launching Puppeteer browser (headless: ${headless})`);

    try {
      const browser = await puppeteer.launch({
        headless,
        args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
      });

      // Handle browser disconnection
      browser.on('disconnected', () => {
        debug('Browser disconnected');
        this.browser = null;
      });

      return browser;
    } catch (error) {
      console.error('Failed to launch browser:', error);
      throw error;
    }
  }

  /**
   * Create a new page with default settings
   */
  public async newPage(): Promise<Page> {
    const browser = await this.getBrowser();
    const page = await browser.newPage();
    
    // Set default timeout
    const timeout = parseInt(process.env.PAGE_TIMEOUT || '30000', 10);
    page.setDefaultTimeout(timeout);
    page.setDefaultNavigationTimeout(timeout);

    // Set a realistic user agent
    await page.setUserAgent(
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    );

    return page;
  }

  /**
   * Close the browser instance
   */
  public async closeBrowser(): Promise<void> {
    if (this.browser) {
      try {
        await this.browser.close();
        this.browser = null;
        debug('Browser closed successfully');
      } catch (error) {
        console.error('Error closing browser:', error);
      }
    }
  }
}

export const browserManager = BrowserManager.getInstance();
