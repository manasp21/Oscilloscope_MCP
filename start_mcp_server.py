#!/usr/bin/env python3
"""
MCP Oscilloscope Server Startup Script

Simple script to start the MCP oscilloscope server with proper configuration.
This script handles environment setup and provides clear startup feedback.

Usage:
    python start_mcp_server.py [--host HOST] [--port PORT] [--log-level LEVEL]
"""

import os
import sys
import argparse
import asyncio
import signal
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from oscilloscope_mcp.mcp_server import OscilloscopeMCPServer
except ImportError as e:
    print(f"❌ Cannot import MCP server: {e}")
    print("Make sure you're running this from the Oscilloscope_MCP directory")
    print("and that all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def setup_signal_handlers(server):
    """Setup graceful shutdown signal handlers."""
    
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        # Create a new event loop for cleanup if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run cleanup
        loop.run_until_complete(server.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        "fastmcp",
        "uvicorn", 
        "starlette",
        "numpy",
        "scipy",
        "structlog",
        "pydantic"
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Missing required dependencies: {', '.join(missing)}")
        print("\nInstall dependencies with:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def print_startup_info(host, port, log_level):
    """Print startup information and usage instructions."""
    print("🔬 MCP Oscilloscope Server")
    print("=" * 50)
    print(f"🌐 Server URL: http://{host}:{port}")
    print(f"📊 Health Check: http://{host}:{port}/health")
    print(f"📡 ADC Data Endpoint: http://{host}:{port}/adc/data")
    print(f"🔌 WebSocket Stream: ws://{host}:{port}/adc/stream")
    print(f"📝 Log Level: {log_level}")
    print()
    print("🎯 Capabilities:")
    print("  ✅ Real ADC data ingestion via HTTP/WebSocket")
    print("  ✅ Professional signal processing pipeline")
    print("  ✅ MCP protocol compliance for AI agents")
    print("  ✅ Real-time measurements and FFT analysis")
    print("  ✅ Protocol decoding (UART, SPI, I2C, CAN)")
    print()
    print("🧪 Testing:")
    print(f"  • Microphone client: python microphone_mcp_client.py --server-url http://{host}:{port}")
    print(f"  • Health check: curl http://{host}:{port}/health")
    print(f"  • MCP tools: Connect with Claude Desktop or VS Code")
    print()
    print("🛑 Stop server: Ctrl+C")
    print("=" * 50)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MCP Oscilloscope Server")
    parser.add_argument("--host", default="localhost", 
                       help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port to bind to (default: 8080)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level (default: INFO)")
    parser.add_argument("--enable-websocket", action="store_true", default=True,
                       help="Enable WebSocket streaming (default: enabled)")
    parser.add_argument("--max-acquisitions", type=int, default=1000,
                       help="Maximum stored acquisitions (default: 1000)")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Set environment variables
    os.environ["MCP_HOST"] = args.host
    os.environ["MCP_PORT"] = str(args.port)
    os.environ["LOG_LEVEL"] = args.log_level
    os.environ["ENABLE_WEBSOCKET"] = str(args.enable_websocket).lower()
    os.environ["MAX_ACQUISITIONS"] = str(args.max_acquisitions)
    
    # Print startup information
    print_startup_info(args.host, args.port, args.log_level)
    
    # Create server
    server = OscilloscopeMCPServer(host=args.host, port=args.port)
    
    # Setup signal handlers
    setup_signal_handlers(server)
    
    try:
        print("🚀 Starting MCP server...")
        await server.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
        
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {args.port} is already in use!")
            print(f"Try a different port: python start_mcp_server.py --port {args.port + 1}")
        else:
            print(f"❌ Network error: {e}")
        return 1
        
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        try:
            await server.stop()
        except:
            pass
        print("✅ Server stopped")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
        sys.exit(0)