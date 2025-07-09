# Windows Setup Guide for Oscilloscope MCP Server

## Overview
This guide will help you set up the Oscilloscope MCP Server on Windows to use your microphone as an analog input source with Claude Desktop.

## Prerequisites
- Windows 10 or 11
- Node.js (version 18 or higher) - [Download here](https://nodejs.org/)
- Claude Desktop - [Download here](https://claude.ai/download)
- Working microphone (built-in or USB)

## Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
npm install
```

### 2. Build the Project
```bash
npm run build
```

### 3. Start the Server
**Option A: Using PowerShell (Recommended)**
```powershell
.\start-mcp-server.ps1
```

**Option B: Using Command Prompt**
```cmd
start-mcp-server.bat
```

### 4. Configure Claude Desktop
1. Open Claude Desktop
2. Navigate to `%APPDATA%\Roaming\Claude\` (Windows + R, then type this path)
3. Create or edit `claude_desktop_config.json`
4. Add this configuration (replace `YOUR_PATH` with actual path):

```json
{
  "mcpServers": {
    "oscilloscope": {
      "command": "node",
      "args": ["C:\\YOUR_PATH\\Oscilloscope_MCP\\dist\\index.js"],
      "env": {
        "HARDWARE_INTERFACE": "microphone",
        "AUDIO_SAMPLE_RATE": "44100",
        "DEBUG": "false",
        "MCP_MODE": "true"
      }
    }
  }
}
```

### 5. Restart Claude Desktop
After saving the configuration, restart Claude Desktop to load the MCP server.

## Testing Your Setup

### 1. Check Hardware Status
In Claude Desktop, ask:
```
Can you check the hardware status of the oscilloscope server?
```

### 2. List Audio Devices
```
Please list the available audio devices for the oscilloscope
```

### 3. Capture Audio from Microphone
```
Can you capture 3 seconds of audio from my microphone and analyze it?
```

### 4. Analyze Audio Spectrum
```
Please analyze the frequency spectrum of the captured audio
```

## Configuration Options

### Hardware Interface Options
- `simulation` - Mock data for testing
- `microphone` - Use computer microphone (default for Windows)
- `usb` - USB-based ADC devices
- `ethernet` - Network-connected ADCs
- `pcie` - PCIe ADC cards

### Audio Sample Rates
- `22050` - Basic quality
- `44100` - CD quality (recommended)
- `48000` - Professional audio
- `96000` - High-resolution audio

### MCP Mode Configuration
- `MCP_MODE` - Controls logging behavior for Claude Desktop integration
  - `"true"` - Uses stderr for logging (required for Claude Desktop)
  - `"false"` - Uses stdout for logging (standalone testing)
  - Auto-detected when running compiled index.js file

### Example Configurations

**Basic Microphone Setup:**
```json
{
  "mcpServers": {
    "oscilloscope": {
      "command": "node",
      "args": ["C:\\Path\\To\\Oscilloscope_MCP\\dist\\index.js"],
      "env": {
        "HARDWARE_INTERFACE": "microphone",
        "AUDIO_SAMPLE_RATE": "44100",
        "MCP_MODE": "true"
      }
    }
  }
}
```

**High-Resolution Audio:**
```json
{
  "mcpServers": {
    "oscilloscope": {
      "command": "node",
      "args": ["C:\\Path\\To\\Oscilloscope_MCP\\dist\\index.js"],
      "env": {
        "HARDWARE_INTERFACE": "microphone",
        "AUDIO_SAMPLE_RATE": "96000",
        "DEBUG": "true",
        "MCP_MODE": "true"
      }
    }
  }
}
```

## Claude Desktop Integration Commands

### Device Management
- `list_audio_devices()` - List available audio devices
- `get_hardware_status()` - Get current configuration
- `configure_hardware()` - Change hardware settings

### Data Acquisition
- `acquire_waveform()` - Capture audio data
- `generate_test_signal()` - Generate test signals
- `get_acquisition_status()` - Check acquisition status

### Analysis Tools
- `measure_parameters()` - Measure signal parameters
- `analyze_spectrum()` - Perform FFT analysis
- `decode_protocol()` - Decode digital protocols

### Example Workflows

**Basic Audio Analysis:**
```
1. configure_hardware(hardware_interface="microphone", audio_sample_rate=44100)
2. acquire_waveform(timeout=5.0, channels=[0])
3. analyze_spectrum(acquisition_id="your_id", window="hamming")
```

**Signal Measurement:**
```
1. acquire_waveform(timeout=3.0)
2. measure_parameters(acquisition_id="your_id", measurements=["frequency", "amplitude", "rms"])
```

## Troubleshooting

### Common Issues

**1. "Node.js not found"**
- Install Node.js from https://nodejs.org/
- Restart command prompt/PowerShell
- Verify with `node --version`

**2. "npm install failed"**
- Run as Administrator
- Clear npm cache: `npm cache clean --force`
- Try: `npm install --force`

**3. "Microphone not working"**
- Check Windows audio settings
- Ensure microphone is not muted
- Try different sample rates
- Enable debug mode: `DEBUG=true`

**4. "Claude Desktop not connecting"**
- Check JSON syntax in config file
- Verify file path is correct
- Use double backslashes in Windows paths
- Restart Claude Desktop

### Debug Mode
Enable debug logging by setting `DEBUG=true` in your configuration:
```json
{
  "env": {
    "DEBUG": "true"
  }
}
```

### Windows-Specific Tips
- Use double backslashes (`\\`) in JSON paths
- Run PowerShell as Administrator if needed
- Check Windows Defender/antivirus settings
- Ensure microphone permissions are enabled

## Advanced Configuration

### Custom Microphone Device
```json
{
  "env": {
    "HARDWARE_INTERFACE": "microphone",
    "MICROPHONE_DEVICE": "USB Audio Device"
  }
}
```

### Multiple Channels
```json
{
  "env": {
    "CHANNELS": "2",
    "AUDIO_SAMPLE_RATE": "48000"
  }
}
```

## Support
- Check the console output for error messages
- Enable debug mode for detailed logging
- Verify microphone permissions in Windows settings
- Test with different audio devices if available

## Next Steps
Once everything is working:
1. Try different signal sources (music, function generator, etc.)
2. Experiment with different analysis parameters
3. Create custom measurement workflows
4. Explore protocol decoding features