import { Page } from 'puppeteer';
import { browserManager } from './browser-manager.js';
import { logger } from './utils/logger.js';

/**
 * Interface for the extracted content from a webpage
 */
export interface ExtractedContent {
  url: string;
  title: string;
  content: string;
  links: { url: string; text: string }[];
  error?: string;
}

/**
 * ContentExtractor handles the webpage content extraction logic
 */
export class ContentExtractor {
  /**
   * Extract content from a URL
   * @param url The URL to extract content from
   * @returns The extracted content
   */
  public static async extractContent(url: string): Promise<ExtractedContent> {
    let page: Page | null = null;
    
    try {
      page = await browserManager.newPage();
      logger.debug(`Navigating to: ${url}`);
      
      // Navigate to the page
      await page.goto(url, { 
        waitUntil: 'networkidle2',
        timeout: parseInt(process.env.PAGE_TIMEOUT || '30000', 10)
      });
      
      // Extract the page title
      const title = await page.title();
      
      // Extract the main content using the content extractor algorithm
      const content = await this.extractMainContent(page);
      
      // Extract relevant links
      const links = await this.extractLinks(page);
      
      return {
        url,
        title,
        content,
        links
      };
    } catch (error) {
      logger.error(`Error extracting content from ${url}:`, error);
      return {
        url,
        title: 'Error',
        content: '',
        links: [],
        error: error instanceof Error ? error.message : String(error)
      };
    } finally {
      if (page) {
        await page.close().catch(err => logger.error('Error closing page:', err));
      }
    }
  }
  
  /**
   * Extract the main content from a page using readability-like heuristics
   * @param page The Puppeteer page object
   * @returns The extracted main content
   */
  private static async extractMainContent(page: Page): Promise<string> {
    return page.evaluate(() => {
      // Define potential content selectors in order of preference
      const contentSelectors = [
        'article',
        'main',
        '.post-content',
        '.entry-content',
        '.content',
        '#content',
        '.post',
        '.article',
        '.blog-post',
      ];
      
      // Define elements to ignore
      const elementsToIgnore = [
        'nav',
        'header',
        'footer',
        'aside',
        '.sidebar',
        '.menu',
        '.navigation',
        '.ads',
        '.advertisement',
        '.related',
        '.comments',
      ];
      
      // Try to find the main content container
      let contentElement: Element | null = null;
      
      // First try with defined selectors
      for (const selector of contentSelectors) {
        const elements = document.querySelectorAll(selector);
        if (elements.length === 1) {
          contentElement = elements[0];
          break;
        } else if (elements.length > 1) {
          // Find the largest element by content length
          let maxLength = 0;
          let maxElement: Element | null = null;
          
          elements.forEach((el) => {
            const length = el.textContent?.length || 0;
            if (length > maxLength) {
              maxLength = length;
              maxElement = el;
            }
          });
          
          if (maxElement) {
            contentElement = maxElement;
            break;
          }
        }
      }
      
      // If no content found, use density analysis to find the main content
      if (!contentElement) {
        const paragraphs = document.querySelectorAll('p');
        if (paragraphs.length > 0) {
          // Find common ancestor with most paragraphs
          const ancestors = new Map<Element, number>();
          
          paragraphs.forEach((p) => {
            let current: Element | null = p;
            while (current && current !== document.body) {
              ancestors.set(current, (ancestors.get(current) || 0) + 1);
              current = current.parentElement;
            }
          });
          
          // Find the element with highest paragraph density
          let maxCount = 0;
          let bestAncestor: Element | null = null;
          
          ancestors.forEach((count, element) => {
            if (count > maxCount) {
              maxCount = count;
              bestAncestor = element;
            }
          });
          
          if (bestAncestor) {
            contentElement = bestAncestor;
          }
        }
      }
      
      // If still no content, use the body
      if (!contentElement) {
        contentElement = document.body;
      }
      
      // Clone the content to avoid modifying the original
      const clonedContent = contentElement.cloneNode(true) as Element;
      
      // Remove ignored elements
      for (const selector of elementsToIgnore) {
        clonedContent.querySelectorAll(selector).forEach(el => el.remove());
      }
      
      // Remove scripts, styles, and other non-content elements
      clonedContent.querySelectorAll('script, style, iframe, noscript').forEach(el => el.remove());
      
      // Extract and format the text content
      const textContent = clonedContent.textContent || '';
      
      // Clean up the content (remove excessive whitespace)
      return textContent
        .replace(/\\s+/g, ' ')
        .replace(/\\n+/g, '\\n')
        .trim();
    });
  }
  
  /**
   * Extract relevant links from the page
   * @param page The Puppeteer page object
   * @returns Array of links with URL and text
   */
  private static async extractLinks(page: Page): Promise<{ url: string; text: string }[]> {
    return page.evaluate(() => {
      const links: { url: string; text: string }[] = [];
      const seen = new Set<string>();
      
      // Get all links from the document
      const anchorElements = document.querySelectorAll('a[href]');
      
      anchorElements.forEach(anchor => {
        const href = anchor.getAttribute('href');
        if (!href) return;
        
        // Create an absolute URL
        const url = new URL(href, window.location.href).toString();
        
        // Skip non-http/https links, anchors, and already seen URLs
        if (!url.startsWith('http') || seen.has(url)) return;
        
        // Skip URLs that likely aren't content pages
        if (
          url.includes('/cdn-cgi/') ||
          url.includes('/wp-admin/') ||
          url.endsWith('.jpg') ||
          url.endsWith('.jpeg') ||
          url.endsWith('.png') ||
          url.endsWith('.gif') ||
          url.endsWith('.pdf') ||
          url.endsWith('.zip')
        ) {
          return;
        }
        
        // Get the link text and clean it
        const text = (anchor.textContent || '')
          .replace(/\\s+/g, ' ')
          .trim();
        
        // Only include links with meaningful text
        if (text.length > 1) {
          links.push({ url, text });
          seen.add(url);
        }
      });
      
      // Return only the most relevant links (max 10)
      return links.slice(0, 10);
    });
  }
}
