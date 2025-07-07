/**
 * Entry point for Oscilloscope MCP Server using Smithery SDK
 */

import { createStatefulServer } from '@smithery/sdk/server/stateful.js';
import createMcpServer, { configSchema } from './server.js';

console.log('🔬 Starting Oscilloscope MCP Server...');

// Create the stateful server using Smithery SDK pattern
const { app } = createStatefulServer(createMcpServer, {
  schema: configSchema
});

// Start the server on the port specified by Smithery (via PORT env var)
const PORT = process.env.PORT || 8081;

app.listen(PORT, () => {
  console.log(`✅ Oscilloscope MCP Server running on port ${PORT}`);
  console.log(`🔧 Session-based configuration supported`);
  console.log(`📡 Ready for ADC hardware integration`);
  console.log(`🌐 MCP endpoint available at /mcp`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down server...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Shutting down server...');
  process.exit(0);
});

// Export for testing
export { createMcpServer, configSchema };