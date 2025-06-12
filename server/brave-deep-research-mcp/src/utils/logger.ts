/**
 * Simple logger utility with debug mode support
 */
class Logger {
  private debugMode: boolean;

  constructor() {
    this.debugMode = process.env.DEBUG_MODE === 'true';
  }

  /**
   * Log a debug message (only in debug mode)
   */
  debug(...args: any[]): void {
    if (this.debugMode) {
      console.error(`[DEBUG] ${new Date().toISOString()}:`, ...args);
    }
  }

  /**
   * Log an info message
   */
  info(...args: any[]): void {
    console.error(`[INFO] ${new Date().toISOString()}:`, ...args);
  }

  /**
   * Log a warning message
   */
  warn(...args: any[]): void {
    console.error(`[WARN] ${new Date().toISOString()}:`, ...args);
  }

  /**
   * Log an error message
   */
  error(...args: any[]): void {
    console.error(`[ERROR] ${new Date().toISOString()}:`, ...args);
  }
}

export const logger = new Logger();
