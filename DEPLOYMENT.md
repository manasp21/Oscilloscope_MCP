# Deployment Guide

## Quick Start

### 1. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run in simulation mode
PYTHONPATH=src python3 -m oscilloscope_mcp.cli serve --interface simulation

# Test functionality
PYTHONPATH=src python3 -m oscilloscope_mcp.cli test
```

### 2. Docker Deployment
```bash
# Build image
docker build -t oscilloscope-mcp .

# Run container
docker run -p 8080:8080 \
  -e HARDWARE_INTERFACE=simulation \
  -e LOG_LEVEL=INFO \
  oscilloscope-mcp
```

### 3. Smithery Deployment

This project is ready for deployment on [Smithery](https://smithery.ai):

1. **Push to Git Repository**
   ```bash
   git add .
   git commit -m "Initial implementation of MCP Oscilloscope Server"
   git push origin main
   ```

2. **Register on Smithery**
   - Visit [Smithery](https://smithery.ai)
   - Add your repository
   - Configure deployment settings

3. **Configure AI Agents**
   - Add the deployed server to Claude Desktop
   - Configure VS Code extension
   - Use in AI workflows

## Environment Variables

- `HARDWARE_INTERFACE`: `simulation`, `usb`, `ethernet`, `pcie` (default: `simulation`)
- `LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)
- `SAMPLE_RATE`: Default sample rate in Hz (default: `100000000`)
- `CHANNELS`: Number of channels (default: `4`)
- `BUFFER_SIZE`: Buffer size in samples (default: `1048576`)
- `TIMEOUT`: Default timeout in seconds (default: `5.0`)

## MCP Client Configuration

### Claude Desktop
Add to your Claude Desktop configuration:
```json
{
  "mcpServers": {
    "oscilloscope": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "oscilloscope-mcp"],
      "env": {
        "HARDWARE_INTERFACE": "simulation"
      }
    }
  }
}
```

### VS Code Extension
Configure the MCP extension to use the deployed server endpoint.

## Performance Tuning

### High-Performance Mode
```bash
# Enable performance optimizations
pip install ".[performance]"

# Run with optimized settings
HARDWARE_INTERFACE=simulation \
SAMPLE_RATE=1000000000 \
BUFFER_SIZE=10485760 \
python3 -m oscilloscope_mcp.cli serve
```

### Hardware Interface
For real hardware, install additional dependencies:
```bash
pip install ".[hardware]"
```

## Monitoring

The server provides structured logging and health endpoints:

- Health check: `GET /health`
- Metrics: Available through structured logs
- Status: Use the self-test CLI command

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH=src
   ```

2. **Permission Errors (Hardware)**
   ```bash
   # Add user to relevant groups
   sudo usermod -a -G dialout $USER  # For serial devices
   sudo usermod -a -G plugdev $USER  # For USB devices
   ```

3. **Performance Issues**
   ```bash
   # Check system resources
   docker stats oscilloscope-mcp
   
   # Adjust memory limits
   docker run -m 2g oscilloscope-mcp
   ```

### Debug Mode
```bash
LOG_LEVEL=DEBUG python3 -m oscilloscope_mcp.cli serve
```