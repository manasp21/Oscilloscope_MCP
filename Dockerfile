# Multi-stage build for optimized production Node.js image
FROM node:18-alpine as builder

# Set build arguments
ARG NODE_ENV=production

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy TypeScript source code
COPY src/ ./src/
COPY tsconfig.json ./

# Install dev dependencies for building
RUN npm ci && npm run build

# Production stage
FROM node:18-alpine as production

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    && rm -rf /var/cache/apk/*

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S oscilloscope -u 1001

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json ./

# Copy built application from builder stage
COPY --from=builder /app/dist ./dist/
COPY --from=builder /app/node_modules ./node_modules/

# Set proper permissions
RUN chown -R oscilloscope:nodejs /app

# Set environment variables for MCP server
ENV NODE_ENV=production \
    LOG_LEVEL=info \
    HARDWARE_INTERFACE=simulation

# Expose MCP server port (Smithery uses PORT env var, defaults to 8081)
EXPOSE 8081

# Switch to non-root user
USER oscilloscope

# Verify installation
RUN node -e "console.log('Node.js version:', process.version); console.log('Application ready');"

# Command to run the server using Smithery SDK pattern
CMD ["node", "dist/index.js"]