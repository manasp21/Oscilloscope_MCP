# Real MCP Oscilloscope Integration Test

This guide demonstrates the **complete end-to-end MCP protocol integration** with real hardware ADC data using microphone input.

## 🎯 What This Tests

✅ **Real MCP Server**: Standalone HTTP server with proper MCP protocol  
✅ **ADC Data Ingestion**: HTTP/WebSocket endpoints for hardware data  
✅ **Signal Processing Pipeline**: Professional oscilloscope algorithms  
✅ **MCP Tool Calls**: Real protocol communication with AI agents  
✅ **Resource Access**: URI-based data retrieval  
✅ **Hardware Integration**: Any ADC source → MCP server pathway  

## 🚀 Quick Test (Windows)

### Terminal 1: Start MCP Server
```powershell
# Navigate to project directory
cd "C:\Users\Manas Pandey\Documents\github\Oscilloscope_MCP"

# Install dependencies (if not done already)
pip install fastmcp uvicorn starlette numpy scipy structlog pydantic aiohttp websockets

# Start the real MCP server
python start_mcp_server.py
```

**Expected Output:**
```
🔬 MCP Oscilloscope Server
==================================================
🌐 Server URL: http://localhost:8080
📊 Health Check: http://localhost:8080/health
📡 ADC Data Endpoint: http://localhost:8080/adc/data
🔌 WebSocket Stream: ws://localhost:8080/adc/stream
📝 Log Level: INFO

🎯 Capabilities:
  ✅ Real ADC data ingestion via HTTP/WebSocket
  ✅ Professional signal processing pipeline
  ✅ MCP protocol compliance for AI agents
  ✅ Real-time measurements and FFT analysis
  ✅ Protocol decoding (UART, SPI, I2C, CAN)

🚀 Starting MCP server...
✅ MCP Server healthy: healthy
```

### Terminal 2: Test with Microphone Client
```powershell
# In a new PowerShell window
cd "C:\Users\Manas Pandey\Documents\github\Oscilloscope_MCP"

# Install audio dependencies
pip install sounddevice matplotlib

# Run microphone ADC client
python microphone_mcp_client.py
```

**Expected Output:**
```
🎤 MCP Oscilloscope Client - Real Hardware Integration Test
======================================================================
📡 Server: http://localhost:8080
🔌 Protocol: HTTP
🎤 Device: microphone_adc_test

✅ MCP Server healthy: healthy

🎤 Available Audio Input Devices:
  0: Microphone (Realtek Audio)
  1: Line In (Built-in)

Select device (0-1, or press Enter for default): [Enter]

🚀 Starting ADC data capture and MCP integration...
💡 Speak into your microphone or play audio
📊 Data will be sent to MCP server for real-time analysis
🔄 Close the plot window to stop

🎵 Audio capture started
📊 Sent acquisition acq_1234567890_abcd1234: 1024 samples
📊 MCP Status: 1 acquisitions
🔧 MCP Measurements: RMS=0.0234, Freq=440.1Hz
📈 MCP Spectrum: Analysis completed for acq_1234567890_abcd1234
```

## 🔧 What's Happening

### 1. Real MCP Server
- **HTTP server** running on localhost:8080
- **FastMCP framework** with proper protocol endpoints
- **ADC data ingestion** via `/adc/data` endpoint
- **WebSocket streaming** via `/adc/stream`
- **Health monitoring** via `/health`

### 2. ADC Data Flow
```
Microphone Audio → ADC Simulation → HTTP POST → MCP Server → Signal Processing → MCP Tools
```

### 3. MCP Protocol Usage
The client demonstrates **real MCP protocol calls**:

```python
# Tool call via HTTP
POST /mcp/call
{
    "method": "tools/call",
    "params": {
        "name": "measure_parameters",
        "arguments": {
            "acquisition_id": "acq_1234567890_abcd1234",
            "measurements": ["rms", "frequency", "amplitude"],
            "channel": 0
        }
    }
}

# Resource access via HTTP  
GET /resources/acquisition/acq_1234567890_abcd1234/waveform
```

### 4. Hardware Integration Ready
The server accepts ADC data from **ANY hardware source**:

```json
POST /adc/data
{
    "device_id": "my_adc_device",
    "sample_rate": 100000.0,
    "timestamp": 1234567890.123,
    "channels": {
        "0": [0.1, 0.2, 0.3, ...],
        "1": [0.4, 0.5, 0.6, ...]
    },
    "metadata": {
        "voltage_range": 3.3,
        "resolution_bits": 12,
        "coupling": "DC"
    }
}
```

## 🧪 Advanced Testing

### Test 1: Manual API Calls
```powershell
# Test health endpoint
curl http://localhost:8080/health

# Send test ADC data
curl -X POST http://localhost:8080/adc/data -H "Content-Type: application/json" -d '{
    "device_id": "test_adc",
    "sample_rate": 1000.0,
    "timestamp": 1234567890,
    "channels": {"0": [0.1, 0.2, 0.3, 0.4, 0.5]},
    "metadata": {"voltage_range": 3.3}
}'
```

### Test 2: WebSocket Streaming
```powershell
# Use WebSocket for real-time streaming
python microphone_mcp_client.py --websocket
```

### Test 3: Multiple Channels
```python
# Modify microphone_mcp_client.py to send multi-channel data
"channels": {
    "0": audio_data_left,
    "1": audio_data_right
}
```

### Test 4: Different ADC Types
```python
# Test with different data formats
from oscilloscope_mcp.hardware.adc_interface import SerialADCInterface

# Serial ADC
adc = SerialADCInterface("test_serial", "/dev/ttyUSB0", 115200)
adc.set_data_format(ADCDataFormat.RAW_BINARY, {
    "bytes_per_sample": 2,
    "num_channels": 4
})
```

## 📊 Success Indicators

✅ **Server Health**: `GET /health` returns `{"status": "healthy"}`  
✅ **Data Ingestion**: ADC data POST returns `{"status": "success", "acquisition_id": "..."}`  
✅ **MCP Tools Work**: Tool calls return valid measurement data  
✅ **Resource Access**: Resource URIs return waveform data  
✅ **Real-time Performance**: <100ms latency for typical operations  
✅ **Protocol Compliance**: Standard MCP format for all endpoints  

## 🔌 AI Agent Integration

Once the server is running, AI agents can connect via standard MCP protocol:

### Claude Desktop Configuration
```json
{
  "mcpServers": {
    "oscilloscope": {
      "command": "python",
      "args": ["start_mcp_server.py"],
      "cwd": "C:\\Users\\Manas Pandey\\Documents\\github\\Oscilloscope_MCP"
    }
  }
}
```

### VS Code Extension
Configure the MCP extension to connect to `http://localhost:8080`

### API Integration
```python
import requests

# AI agent makes MCP tool calls
response = requests.post("http://localhost:8080/mcp/call", json={
    "method": "tools/call", 
    "params": {
        "name": "acquire_waveform",
        "arguments": {"timeout": 5.0}
    }
})
```

## 🚨 Troubleshooting

### Server Won't Start
```
❌ Port 8080 is already in use!
```
**Solution**: Use different port
```powershell
python start_mcp_server.py --port 8081
```

### Client Can't Connect
```
❌ Cannot reach MCP server: Connection refused
```
**Solution**: Ensure server is running and check firewall

### Audio Issues
```
❌ No audio input devices found!
```
**Solution**: Check microphone permissions and drivers

### Import Errors
```
❌ Cannot import MCP modules
```
**Solution**: Install dependencies
```powershell
pip install -r requirements.txt
```

## 🎯 What This Proves

✅ **Real MCP Server**: Not simulation - actual HTTP server with MCP protocol  
✅ **Hardware Ready**: Can accept data from any ADC via standardized interface  
✅ **AI Agent Compatible**: Ready for Claude Desktop, VS Code, and other MCP clients  
✅ **Production Quality**: Professional signal processing with real-time performance  
✅ **Extensible**: Easy to add new ADC hardware types and analysis functions  

This demonstrates a **complete, working MCP oscilloscope server** that can interface with real hardware and provide professional signal analysis capabilities to AI agents!