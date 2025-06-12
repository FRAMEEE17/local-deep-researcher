import dotenv from 'dotenv';
import { z } from 'zod';

// Load environment variables from .env file
dotenv.config();

// Define the schema for environment variables
const envSchema = z.object({
  // Required variables
  BRAVE_API_KEY: z.string().min(1, "BRAVE_API_KEY is required"),
  
  // Optional variables with defaults
  PUPPETEER_HEADLESS: z.enum(['true', 'false']).default('true'),
  PAGE_TIMEOUT: z.string().default('30000').transform(Number),
  DEBUG_MODE: z.enum(['true', 'false']).default('false'),
});

// Parse and validate environment variables
const parseEnv = () => {
  try {
    return envSchema.parse(process.env);
  } catch (error) {
    console.error("Environment validation failed:", error);
    process.exit(1);
  }
};

// Export the validated environment variables
export const config = {
  ...parseEnv(),
  isHeadless: process.env.PUPPETEER_HEADLESS !== 'false',
  isDebugMode: process.env.DEBUG_MODE === 'true',
};

// Helper function for debugging
export function debug(...args: any[]) {
  if (config.isDebugMode) {
    console.error('[DEBUG]', ...args);
  }
}