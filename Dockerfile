# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    libfftw3-dev \
    libblas-dev \
    liblapack-dev \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source code
COPY . /app
WORKDIR /app

# Install the package (normal install, not editable)
RUN pip install .

# Verify package installation
RUN python -c "import oscilloscope_mcp; print(f'Package installed: {oscilloscope_mcp.__version__}')"

# Production stage
FROM python:3.11-slim as production

# Set runtime arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libfftw3-3 \
    libblas3 \
    liblapack3 \
    libusb-1.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r oscilloscope && useradd -r -g oscilloscope oscilloscope

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app /app
WORKDIR /app

# Set proper permissions
RUN chown -R oscilloscope:oscilloscope /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    LOG_LEVEL=INFO \
    HARDWARE_INTERFACE=simulation

# Expose health check port
EXPOSE 8080

# Verify package is available in production stage
RUN python -c "import oscilloscope_mcp; print(f'Production stage - Package available: {oscilloscope_mcp.__version__}')"

# Switch to non-root user
USER oscilloscope

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f --max-time 8 http://localhost:8080/health | grep -q "healthy" || exit 1

# Command to run the server
CMD ["sh", "-c", "echo 'Starting Oscilloscope MCP Server...' && python -c 'import oscilloscope_mcp; print(f\"Module check OK: v{oscilloscope_mcp.__version__}\")' && python -m oscilloscope_mcp.cli serve --host 0.0.0.0 --port 8080"]