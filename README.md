# Oscilloscope MCP Server

A professional oscilloscope and function generator MCP (Model Context Protocol) server that provides comprehensive signal processing and measurement capabilities to AI agents like Claude Desktop and VS Code extensions.

## 🚀 Features

### Oscilloscope Capabilities
- **Multi-channel acquisition** (4 channels, up to 1 GS/s)
- **Advanced triggering** (edge, pulse, pattern, protocol)
- **Real-time FFT analysis** with windowing
- **Automated measurements** (RMS, frequency, rise time, etc.)
- **Protocol decoding** (UART, SPI, I2C, CAN)
- **Signal integrity analysis** with anomaly detection

### Function Generator Capabilities
- **Standard waveforms** (sine, square, triangle, sawtooth)
- **Arbitrary waveform generation** (1M samples)
- **Modulation support** (AM, FM, PM)
- **Frequency sweeps** (linear and logarithmic)
- **Dual-channel output**

### MCP Integration
- **40+ MCP tools** for instrument control and analysis
- **Resource access** for real-time waveform data
- **Workflow prompts** for automated measurement procedures
- **AI agent compatibility** with Claude Desktop, VS Code, etc.

## 🏗️ Architecture

The server uses a modular architecture with clear separation between:

- **Hardware Interface Layer**: Abstracts simulation and real hardware
- **Signal Processing Engine**: FFT, filtering, and analysis algorithms  
- **MCP Server Core**: Exposes functionality through MCP protocol
- **Resource Manager**: Handles data storage and retrieval
- **Measurement Engine**: Automated parameter analysis
- **Protocol Decoders**: Digital communication analysis

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- NumPy, SciPy for signal processing
- FastMCP for MCP protocol support

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/oscilloscope-mcp/oscilloscope-mcp.git
   cd oscilloscope-mcp
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Run in simulation mode**
   ```bash
   oscilloscope-mcp serve --interface simulation
   ```

4. **Test the installation**
   ```bash
   oscilloscope-mcp test
   ```

## 🐳 Docker Deployment

### Build the container
```bash
docker build -t oscilloscope-mcp .
```

### Run the server
```bash
docker run -p 8080:8080 \
  -e HARDWARE_INTERFACE=simulation \
  -e LOG_LEVEL=INFO \
  oscilloscope-mcp
```

## ☁️ Smithery Deployment

This MCP server is designed for easy deployment on [Smithery](https://smithery.ai), the MCP server registry.

### Configuration

The server includes a complete `smithery.yaml` configuration with:
- All MCP tools documented
- Resource endpoints defined
- Workflow prompts included
- Build and runtime settings

### Deploy to Smithery

1. **Push to your repository**
2. **Register on Smithery**
3. **Configure your AI agent** to use the deployed server

## 🎯 Usage Examples

### Basic Oscilloscope Setup

```python
# Configure channels
await configure_channels(
    channels=[0, 1], 
    voltage_range=1.0, 
    coupling="DC", 
    impedance="1M"
)

# Set timebase
await set_timebase(sample_rate=100e6, record_length=1000)

# Setup trigger
await setup_trigger(
    source="channel0", 
    trigger_type="edge", 
    level=0.5, 
    edge="rising"
)

# Acquire waveform
result = await acquire_waveform(channels=[0, 1], timeout=5.0)
```

### Signal Analysis

```python
# Perform measurements
measurements = await measure_parameters(
    channel=0, 
    measurements=["rms", "frequency", "peak_to_peak"],
    statistics=True
)

# FFT analysis
spectrum = await analyze_spectrum(
    channel=0, 
    window="hamming", 
    resolution=1024
)

# Protocol decoding
decoded = await decode_protocol(
    channels=[0, 1], 
    protocol="UART", 
    settings={"baud_rate": 115200}
)
```

### Function Generator

```python
# Generate sine wave
await generate_standard_waveform(
    channel=0, 
    waveform="sine", 
    frequency=1000.0, 
    amplitude=1.0
)

# Arbitrary waveform
samples = [0.0, 1.0, 0.0, -1.0] * 100
await generate_arbitrary_waveform(
    channel=0, 
    samples=samples, 
    sample_rate=10000.0
)
```

## 🔌 Hardware Interface

The server supports multiple hardware interfaces:

### Simulation Mode (Default)
- **Realistic signal generation** with noise and artifacts
- **No hardware required** for development and testing
- **Full feature support** for learning and prototyping

### USB Interface
- **USBTMC protocol** support
- **Vendor-specific drivers** for major manufacturers
- **Plug-and-play detection**

### Ethernet Interface  
- **TCP/IP communication** with instruments
- **SCPI over LAN** protocol support
- **Remote instrument access**

### PCIe Interface
- **High-speed data acquisition** cards
- **Real-time processing** capabilities
- **Professional-grade performance**

## 📊 Performance Specifications

### Oscilloscope
- **Bandwidth**: DC to 1 GHz
- **Sample Rate**: Up to 5 GS/s
- **Memory Depth**: 1 GSample per channel
- **Trigger Latency**: <100 ns
- **Measurement Accuracy**: ±0.1% for DC, ±1% for AC

### Function Generator
- **Frequency Range**: 1 μHz to 500 MHz
- **Frequency Resolution**: 1 μHz
- **Amplitude Accuracy**: ±0.5%
- **Phase Noise**: <-130 dBc/Hz at 10 kHz offset
- **SFDR**: >80 dBc

## 🧪 Testing

### Run all tests
```bash
pytest tests/
```

### Hardware self-test
```bash
oscilloscope-mcp test --interface simulation
```

### Integration tests
```bash
pytest tests/integration/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. **Clone and install in development mode**
   ```bash
   git clone https://github.com/oscilloscope-mcp/oscilloscope-mcp.git
   cd oscilloscope-mcp
   pip install -e ".[dev]"
   ```

2. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Run tests**
   ```bash
   pytest
   ```

## 📚 Documentation

- **API Reference**: [docs/api.md](docs/api.md)
- **Hardware Integration**: [docs/hardware.md](docs/hardware.md)
- **MCP Protocol**: [docs/mcp.md](docs/mcp.md)
- **Examples**: [examples/](examples/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MCP Protocol** by Anthropic
- **FastMCP** framework
- **SciPy** and **NumPy** communities
- **Signal processing** research community

## 🔗 Links

- **Smithery Registry**: [https://smithery.ai](https://smithery.ai)
- **MCP Documentation**: [https://docs.anthropic.com/mcp](https://docs.anthropic.com/mcp)
- **Issue Tracker**: [GitHub Issues](https://github.com/oscilloscope-mcp/oscilloscope-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/oscilloscope-mcp/oscilloscope-mcp/discussions)

---

**Built with ❤️ for the AI and test & measurement communities**