# Oscilloscope MCP Server with Microphone Integration

A professional oscilloscope and function generator MCP (Model Context Protocol) server that provides comprehensive signal processing and measurement capabilities to AI agents like Claude Desktop. **Now with Windows microphone support for real-time audio analysis!**

## Features

### Oscilloscope Capabilities
- **Multi-channel acquisition** (4 channels, configurable sample rates)
- **Real-time microphone input** for Windows audio analysis
- **Advanced FFT analysis** with windowing functions
- **Automated measurements** (RMS, frequency, amplitude, etc.)
- **Protocol decoding** (UART, SPI, I2C, CAN)
- **Signal processing** with spectrum analysis

### Function Generator Capabilities
- **Standard waveforms** (sine, square, triangle, sawtooth, noise)
- **Configurable parameters** (frequency, amplitude, duration)
- **Test signal injection** for calibration and testing
- **Dual-channel output simulation**

### MCP Integration
- **9 MCP tools** for instrument control and analysis
- **3 resource endpoints** for real-time data access
- **2 workflow prompts** for guided setup
- **Claude Desktop compatibility** with Windows integration

### Hardware Interface Support
- **Simulation mode** - Mock data for development
- **Microphone mode** - Real-time Windows audio capture
- **USB/Ethernet/PCIe** - Professional ADC hardware support

## Architecture

The server uses a modular TypeScript architecture with:

- **Hardware Interface Layer**: Abstracts simulation and real hardware
- **Signal Processing Engine**: FFT, filtering, and analysis algorithms  
- **MCP Server Core**: Exposes functionality through MCP protocol
- **Data Store**: Handles acquisition storage and caching
- **Measurement Engine**: Automated parameter analysis
- **Protocol Decoders**: Digital communication analysis

## Quick Start (Windows + Claude Desktop)

### Prerequisites
- Windows 10/11 with working microphone
- [Node.js](https://nodejs.org/) (version 18 or higher)
- [Claude Desktop](https://claude.ai/download)

### Installation
1. **Clone and install**
   ```bash
   git clone https://github.com/oscilloscope-mcp/oscilloscope-mcp.git
   cd oscilloscope-mcp
   npm install
   ```

2. **Build the project**
   ```bash
   npm run build
   ```

3. **Start the server**
   ```powershell
   .\start-mcp-server.ps1
   ```

4. **Configure Claude Desktop**
   - Open `%APPDATA%\Roaming\Claude\claude_desktop_config.json`
   - Add the server configuration (see WINDOWS_SETUP.md)

### Testing Your Setup
In Claude Desktop, try these commands:
- "Check the hardware status of the oscilloscope server"
- "List available audio devices"
- "Capture 3 seconds of audio from my microphone and analyze it"

## MCP Tools Available

### Oscilloscope Tools
- `get_acquisition_status` - Check current acquisition status
- `acquire_waveform` - Capture data from microphone or simulation
- `measure_parameters` - Automated signal measurements
- `analyze_spectrum` - FFT analysis with windowing

### Function Generator Tools
- `generate_test_signal` - Create calibration signals

### Device Management Tools
- `list_audio_devices` - Enumerate available audio devices
- `configure_hardware` - Change hardware settings
- `get_hardware_status` - Get current configuration

### Analysis Tools
- `decode_protocol` - Decode digital communication protocols

## Usage Examples

### Basic Microphone Analysis
```
1. configure_hardware(hardware_interface="microphone", audio_sample_rate=44100)
2. acquire_waveform(timeout=5.0, channels=[0])
3. measure_parameters(acquisition_id="...", measurements=["frequency", "amplitude", "rms"])
4. analyze_spectrum(acquisition_id="...", window="hamming")
```

### Device Selection
```
1. list_audio_devices()
2. configure_hardware(hardware_interface="microphone", microphone_device="USB Audio")
3. get_hardware_status()
```

### Signal Generation and Testing
```
1. generate_test_signal(signal_type="sine", frequency=1000, amplitude=1.0)
2. analyze_spectrum(acquisition_id="test_...", resolution=1024)
```

## Hardware Interface Configuration

### Microphone Mode (Default for Windows)
- **Real-time audio capture** from system microphone
- **Configurable sample rates** (22050, 44100, 48000, 96000 Hz)
- **Device selection** from available audio inputs
- **Multi-channel simulation** from mono input

### Simulation Mode
- **Realistic signal generation** with noise
- **No hardware required** for development
- **Full feature support** for testing

### Professional Hardware
- **USB/Ethernet/PCIe** support for ADC devices
- **High-speed data acquisition** capabilities
- **Real-time processing** support

## Configuration Options

### Hardware Interface Types
```typescript
- "simulation" - Mock data for testing
- "microphone" - Windows audio capture
- "usb" - USB-based ADC devices
- "ethernet" - Network-connected ADCs
- "pcie" - PCIe ADC cards
```

### Audio Sample Rates
```typescript
- 22050 - Basic quality
- 44100 - CD quality (recommended)
- 48000 - Professional audio
- 96000 - High-resolution audio
```

### Environment Variables
```bash
HARDWARE_INTERFACE=microphone
AUDIO_SAMPLE_RATE=44100
DEBUG=false
MICROPHONE_DEVICE=default
```

## Testing

### Run TypeScript compilation
```bash
npm run build
```

### Start development server
```bash
npm run dev
```

### Test with Claude Desktop
See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing scenarios.

## Windows Integration

### Automatic Setup
- **PowerShell script** for easy server startup
- **Batch file** for command prompt users
- **Claude Desktop configuration** templates
- **Device enumeration** for audio selection

### Audio Device Support
- **Built-in microphones** and line inputs
- **USB audio devices** and interfaces
- **Professional audio equipment**
- **Multiple device switching**

## Documentation

- **[Windows Setup Guide](WINDOWS_SETUP.md)** - Complete Windows installation
- **[Testing Guide](TESTING_GUIDE.md)** - Comprehensive testing scenarios
- **[API Reference](src/index.ts)** - MCP tools and configuration
- **[Hardware Integration](smithery.yaml)** - Smithery deployment config

## Development

### Local Development
```bash
npm install
npm run dev
```

### Build for Production
```bash
npm run build
npm run start
```

### Smithery Deployment
```bash
npm run smithery:build
npm run smithery:dev
```

## Project Structure

```
oscilloscope-mcp/
├── src/
│   ├── index.ts              # Main MCP server implementation
│   ├── tools/                # MCP tool implementations
│   ├── resources/            # MCP resource handlers
│   └── prompts/              # Workflow prompts
├── dist/                     # Compiled TypeScript output
├── start-mcp-server.ps1      # Windows PowerShell startup script
├── start-mcp-server.bat      # Windows batch startup script
├── WINDOWS_SETUP.md          # Windows setup instructions
├── TESTING_GUIDE.md          # Testing scenarios
└── smithery.yaml             # Smithery deployment config
```

## Contributing

We welcome contributions! Please:

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests** for new functionality
4. **Update documentation**
5. **Submit a pull request**

### Development Setup
```bash
git clone https://github.com/oscilloscope-mcp/oscilloscope-mcp.git
cd oscilloscope-mcp
npm install
npm run build
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MCP Protocol** by Anthropic
- **@modelcontextprotocol/sdk** for TypeScript implementation
- **node-microphone** and **sox-stream** for Windows audio capture
- **Claude Desktop** for AI integration
- **Signal processing** research community

## Links

- **Claude Desktop**: [https://claude.ai/download](https://claude.ai/download)
- **MCP Documentation**: [https://docs.anthropic.com/mcp](https://docs.anthropic.com/mcp)
- **Smithery Registry**: [https://smithery.ai](https://smithery.ai)
- **Node.js**: [https://nodejs.org/](https://nodejs.org/)

---

**Built with professional focus for AI-powered signal analysis and test & measurement**

### Quick Links
- [Windows Setup](WINDOWS_SETUP.md)
- [Testing Guide](TESTING_GUIDE.md)
- [API Reference](src/index.ts)
- [Smithery Config](smithery.yaml)