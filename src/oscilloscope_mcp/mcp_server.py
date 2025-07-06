"""
Standalone MCP Server for Oscilloscope and Function Generator.

This is a real HTTP-based MCP server that accepts ADC data from external hardware
sources and processes it through professional signal processing algorithms.

The server provides:
- Real MCP protocol endpoints (tools, resources, prompts)
- ADC data ingestion via HTTP/WebSocket
- Real-time signal processing pipeline
- Professional oscilloscope and function generator capabilities
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import traceback

import numpy as np
import structlog
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket

from .signal_processing.engine import SignalProcessor
from .measurements.analyzer import MeasurementAnalyzer
from .protocols.decoder import ProtocolDecoder


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# Data Models for ADC Interface
class ADCDataPoint(BaseModel):
    """Single ADC data point."""
    timestamp: float
    channel: int
    value: float
    voltage: Optional[float] = None
    

class ADCDataBatch(BaseModel):
    """Batch of ADC data points."""
    device_id: str
    sample_rate: float
    timestamp: float
    channels: Dict[int, List[float]]
    metadata: Optional[Dict[str, Any]] = {}


class MeasurementRequest(BaseModel):
    """Request for signal measurements."""
    acquisition_id: str
    measurements: List[str]
    channel: Optional[int] = 0
    statistics: Optional[bool] = False


class SpectrumRequest(BaseModel):
    """Request for spectrum analysis."""
    acquisition_id: str
    channel: Optional[int] = 0
    window: Optional[str] = "hamming"
    resolution: Optional[int] = 1024


# Real-time Data Storage
class DataStore:
    """Thread-safe data storage for real-time ADC data."""
    
    def __init__(self):
        self.acquisitions = {}
        self.latest_data = {}
        self.spectrum_cache = {}
        self.measurement_cache = {}
        self._lock = asyncio.Lock()
    
    async def store_adc_data(self, data: ADCDataBatch) -> str:
        """Store ADC data batch and return acquisition ID."""
        acquisition_id = f"acq_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        async with self._lock:
            # Convert to standard format
            waveform_data = {
                "time": np.arange(len(list(data.channels.values())[0])) / data.sample_rate,
                "channels": {},
                "sample_rate": data.sample_rate,
                "timestamp": data.timestamp,
                "device_id": data.device_id,
                "acquisition_id": acquisition_id
            }
            
            for ch_num, ch_data in data.channels.items():
                waveform_data["channels"][ch_num] = {
                    "voltage": ch_data,
                    "voltage_range": data.metadata.get("voltage_range", 1.0),
                    "coupling": data.metadata.get("coupling", "DC"),
                    "impedance": data.metadata.get("impedance", "1M")
                }
            
            # Store full acquisition
            self.acquisitions[acquisition_id] = waveform_data
            
            # Update latest data for each channel
            for ch_num in data.channels.keys():
                self.latest_data[ch_num] = acquisition_id
        
        logger.info("ADC data stored", 
                   acquisition_id=acquisition_id,
                   device_id=data.device_id,
                   channels=list(data.channels.keys()),
                   samples=len(list(data.channels.values())[0]))
        
        return acquisition_id
    
    async def get_acquisition(self, acquisition_id: str) -> Optional[Dict[str, Any]]:
        """Get acquisition data by ID."""
        async with self._lock:
            return self.acquisitions.get(acquisition_id)
    
    async def get_latest_for_channel(self, channel: int) -> Optional[Dict[str, Any]]:
        """Get latest acquisition data for a specific channel."""
        async with self._lock:
            if channel in self.latest_data:
                latest_id = self.latest_data[channel]
                return self.acquisitions.get(latest_id)
            return None
    
    async def cache_spectrum(self, acquisition_id: str, spectrum_data: Dict[str, Any]) -> None:
        """Cache spectrum analysis results."""
        async with self._lock:
            self.spectrum_cache[acquisition_id] = spectrum_data
    
    async def get_cached_spectrum(self, acquisition_id: str) -> Optional[Dict[str, Any]]:
        """Get cached spectrum data."""
        async with self._lock:
            return self.spectrum_cache.get(acquisition_id)
    
    async def cache_measurements(self, acquisition_id: str, measurements: Dict[str, Any]) -> None:
        """Cache measurement results."""
        async with self._lock:
            self.measurement_cache[acquisition_id] = measurements
    
    async def get_cached_measurements(self, acquisition_id: str) -> Optional[Dict[str, Any]]:
        """Get cached measurement data."""
        async with self._lock:
            return self.measurement_cache.get(acquisition_id)


class OscilloscopeMCPServer:
    """Real MCP server for oscilloscope and function generator."""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        
        # Initialize MCP server
        self.mcp = FastMCP("oscilloscope-function-generator")
        
        # Initialize processing components
        self.signal_processor = SignalProcessor()
        self.measurement_analyzer = MeasurementAnalyzer()
        self.protocol_decoder = ProtocolDecoder()
        
        # Real-time data storage
        self.data_store = DataStore()
        
        # Server state
        self.is_running = False
        self.connected_clients = set()
        
        # Configuration
        self.config = self._load_config()
        
        # Register MCP endpoints
        self._register_oscilloscope_tools()
        self._register_function_generator_tools()
        self._register_analysis_tools()
        self._register_resources()
        self._register_prompts()
        
        # Create additional HTTP routes for ADC data ingestion
        self._setup_adc_routes()
        
        logger.info("MCP server initialized", host=host, port=port, config=self.config)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load server configuration from environment variables."""
        return {
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "max_acquisitions": int(os.getenv("MAX_ACQUISITIONS", "1000")),
            "max_buffer_size": int(os.getenv("MAX_BUFFER_SIZE", "10485760")),  # 10MB
            "enable_websocket": os.getenv("ENABLE_WEBSOCKET", "true").lower() == "true",
            "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
        }
    
    def _setup_adc_routes(self):
        """Setup additional HTTP routes for ADC data ingestion."""
        
        async def health_check(request):
            """Health check endpoint."""
            return JSONResponse({
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "acquisitions_count": len(self.data_store.acquisitions),
                "connected_clients": len(self.connected_clients)
            })
        
        async def adc_data_ingestion(request):
            """HTTP endpoint for ADC data ingestion."""
            try:
                data_json = await request.json()
                adc_data = ADCDataBatch(**data_json)
                
                # Store the data
                acquisition_id = await self.data_store.store_adc_data(adc_data)
                
                return JSONResponse({
                    "status": "success",
                    "acquisition_id": acquisition_id,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                logger.error("ADC data ingestion failed", error=str(e))
                return JSONResponse({
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }, status_code=400)
        
        async def websocket_adc_endpoint(websocket):
            """WebSocket endpoint for real-time ADC data streaming."""
            await websocket.accept()
            client_id = f"client_{uuid.uuid4().hex[:8]}"
            self.connected_clients.add(client_id)
            
            logger.info("WebSocket client connected", client_id=client_id)
            
            try:
                while True:
                    # Receive ADC data via WebSocket
                    data_json = await websocket.receive_json()
                    adc_data = ADCDataBatch(**data_json)
                    
                    # Store the data
                    acquisition_id = await self.data_store.store_adc_data(adc_data)
                    
                    # Send confirmation back
                    await websocket.send_json({
                        "status": "received",
                        "acquisition_id": acquisition_id,
                        "timestamp": time.time()
                    })
                    
            except Exception as e:
                logger.error("WebSocket error", client_id=client_id, error=str(e))
            finally:
                self.connected_clients.discard(client_id)
                logger.info("WebSocket client disconnected", client_id=client_id)
        
        # Add routes to FastMCP app
        self.additional_routes = [
            Route("/health", health_check, methods=["GET"]),
            Route("/adc/data", adc_data_ingestion, methods=["POST"]),
        ]
        
        if self.config["enable_websocket"]:
            self.additional_routes.append(
                WebSocketRoute("/adc/stream", websocket_adc_endpoint)
            )
    
    def _register_oscilloscope_tools(self):
        """Register oscilloscope MCP tools."""
        
        @self.mcp.tool
        async def get_acquisition_status() -> Dict[str, Any]:
            """Get current acquisition status and available data."""
            acquisitions = list(self.data_store.acquisitions.keys())
            latest_data = dict(self.data_store.latest_data)
            
            return {
                "status": "success",
                "total_acquisitions": len(acquisitions),
                "recent_acquisitions": acquisitions[-10:] if acquisitions else [],
                "latest_by_channel": latest_data,
                "connected_clients": len(self.connected_clients),
                "timestamp": time.time()
            }
        
        @self.mcp.tool
        async def acquire_waveform(
            timeout: float = 5.0,
            channels: Optional[List[int]] = None
        ) -> Dict[str, Any]:
            """Wait for new ADC data acquisition."""
            start_time = time.time()
            initial_count = len(self.data_store.acquisitions)
            
            # Wait for new data
            while time.time() - start_time < timeout:
                current_count = len(self.data_store.acquisitions)
                if current_count > initial_count:
                    # New data available
                    acquisition_ids = list(self.data_store.acquisitions.keys())
                    latest_id = acquisition_ids[-1]
                    
                    return {
                        "status": "success",
                        "acquisition_id": latest_id,
                        "timestamp": time.time(),
                        "channels_available": list(self.data_store.acquisitions[latest_id]["channels"].keys())
                    }
                
                await asyncio.sleep(0.1)  # Check every 100ms
            
            return {
                "status": "timeout",
                "error": f"No new data received within {timeout} seconds",
                "timestamp": time.time()
            }
        
        @self.mcp.tool
        async def measure_parameters(
            acquisition_id: str,
            measurements: List[str],
            channel: int = 0,
            statistics: bool = False
        ) -> Dict[str, Any]:
            """Perform automated measurements on acquired data."""
            try:
                # Check cache first
                cached = await self.data_store.get_cached_measurements(acquisition_id)
                cache_key = f"{channel}_{'+'.join(sorted(measurements))}_{statistics}"
                
                if cached and cache_key in cached:
                    logger.info("Returning cached measurements", acquisition_id=acquisition_id)
                    return cached[cache_key]
                
                # Get acquisition data
                waveform_data = await self.data_store.get_acquisition(acquisition_id)
                if not waveform_data:
                    return {
                        "status": "error",
                        "error": f"Acquisition {acquisition_id} not found",
                        "timestamp": time.time()
                    }
                
                if channel not in waveform_data["channels"]:
                    return {
                        "status": "error",
                        "error": f"Channel {channel} not found in acquisition",
                        "available_channels": list(waveform_data["channels"].keys()),
                        "timestamp": time.time()
                    }
                
                # Perform measurements
                results = await self.measurement_analyzer.measure_parameters(
                    waveform_data=waveform_data,
                    measurements=measurements,
                    statistics=statistics
                )
                
                # Cache results
                if not cached:
                    cached = {}
                cached[cache_key] = {
                    "status": "success",
                    "acquisition_id": acquisition_id,
                    "channel": channel,
                    "measurements": measurements,
                    "results": results,
                    "timestamp": time.time()
                }
                
                await self.data_store.cache_measurements(acquisition_id, cached)
                
                return cached[cache_key]
                
            except Exception as e:
                logger.error("Measurement failed", error=str(e), acquisition_id=acquisition_id)
                return {
                    "status": "error",
                    "error": str(e),
                    "acquisition_id": acquisition_id,
                    "timestamp": time.time()
                }
        
        @self.mcp.tool
        async def analyze_spectrum(
            acquisition_id: str,
            channel: int = 0,
            window: str = "hamming",
            resolution: int = 1024
        ) -> Dict[str, Any]:
            """Perform FFT analysis on acquired data."""
            try:
                # Check cache first
                cached = await self.data_store.get_cached_spectrum(acquisition_id)
                cache_key = f"{channel}_{window}_{resolution}"
                
                if cached and cache_key in cached:
                    logger.info("Returning cached spectrum", acquisition_id=acquisition_id)
                    return cached[cache_key]
                
                # Get acquisition data
                waveform_data = await self.data_store.get_acquisition(acquisition_id)
                if not waveform_data:
                    return {
                        "status": "error",
                        "error": f"Acquisition {acquisition_id} not found",
                        "timestamp": time.time()
                    }
                
                if channel not in waveform_data["channels"]:
                    return {
                        "status": "error",
                        "error": f"Channel {channel} not found in acquisition",
                        "available_channels": list(waveform_data["channels"].keys()),
                        "timestamp": time.time()
                    }
                
                # Perform spectrum analysis
                spectrum_data = await self.signal_processor.analyze_spectrum(
                    waveform_data=waveform_data,
                    window=window,
                    resolution=resolution
                )
                
                # Cache results
                if not cached:
                    cached = {}
                    
                cached[cache_key] = {
                    "status": "success",
                    "acquisition_id": acquisition_id,
                    "channel": channel,
                    "window": window,
                    "resolution": resolution,
                    "spectrum": spectrum_data,
                    "timestamp": time.time()
                }
                
                await self.data_store.cache_spectrum(acquisition_id, cached)
                
                return cached[cache_key]
                
            except Exception as e:
                logger.error("Spectrum analysis failed", error=str(e), acquisition_id=acquisition_id)
                return {
                    "status": "error",
                    "error": str(e),
                    "acquisition_id": acquisition_id,
                    "timestamp": time.time()
                }
    
    def _register_function_generator_tools(self):
        """Register function generator MCP tools."""
        
        @self.mcp.tool
        async def generate_test_signal(
            signal_type: str = "sine",
            frequency: float = 1000.0,
            amplitude: float = 1.0,
            sample_rate: float = 44100.0,
            duration: float = 1.0,
            channel: int = 0
        ) -> Dict[str, Any]:
            """Generate a test signal and inject it as ADC data."""
            try:
                # Generate time axis
                t = np.linspace(0, duration, int(sample_rate * duration))
                
                # Generate waveform
                if signal_type == "sine":
                    signal_data = amplitude * np.sin(2 * np.pi * frequency * t)
                elif signal_type == "square":
                    signal_data = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
                elif signal_type == "triangle":
                    signal_data = amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t))
                elif signal_type == "sawtooth":
                    signal_data = amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
                elif signal_type == "noise":
                    signal_data = amplitude * np.random.normal(0, 1, len(t))
                else:
                    return {
                        "status": "error",
                        "error": f"Unknown signal type: {signal_type}",
                        "timestamp": time.time()
                    }
                
                # Create ADC data batch
                adc_data = ADCDataBatch(
                    device_id="function_generator",
                    sample_rate=sample_rate,
                    timestamp=time.time(),
                    channels={channel: signal_data.tolist()},
                    metadata={
                        "signal_type": signal_type,
                        "frequency": frequency,
                        "amplitude": amplitude,
                        "generated": True
                    }
                )
                
                # Store as if it came from ADC
                acquisition_id = await self.data_store.store_adc_data(adc_data)
                
                return {
                    "status": "success",
                    "signal_type": signal_type,
                    "frequency": frequency,
                    "amplitude": amplitude,
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "samples": len(signal_data),
                    "acquisition_id": acquisition_id,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error("Test signal generation failed", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
    
    def _register_analysis_tools(self):
        """Register advanced analysis tools."""
        
        @self.mcp.tool
        async def decode_protocol(
            acquisition_id: str,
            protocol: str,
            settings: Dict[str, Any],
            channels: List[int]
        ) -> Dict[str, Any]:
            """Decode digital communication protocols from acquired data."""
            try:
                waveform_data = await self.data_store.get_acquisition(acquisition_id)
                if not waveform_data:
                    return {
                        "status": "error",
                        "error": f"Acquisition {acquisition_id} not found",
                        "timestamp": time.time()
                    }
                
                # Extract data for specified channels
                channel_data = {}
                for ch in channels:
                    if ch in waveform_data["channels"]:
                        channel_data[ch] = waveform_data["channels"][ch]
                
                if not channel_data:
                    return {
                        "status": "error",
                        "error": "No valid channels found for protocol decoding",
                        "requested_channels": channels,
                        "available_channels": list(waveform_data["channels"].keys()),
                        "timestamp": time.time()
                    }
                
                # Perform protocol decoding
                decoded_data = await self.protocol_decoder.decode_protocol(
                    waveform_data={"channels": channel_data},
                    protocol=protocol,
                    settings=settings
                )
                
                return {
                    "status": "success",
                    "acquisition_id": acquisition_id,
                    "protocol": protocol,
                    "channels": channels,
                    "decoded_data": decoded_data,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error("Protocol decoding failed", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "acquisition_id": acquisition_id,
                    "protocol": protocol,
                    "timestamp": time.time()
                }
    
    def _register_resources(self):
        """Register MCP resources for data access."""
        
        @self.mcp.resource("acquisition://{acquisition_id}/waveform")
        async def get_waveform_data(acquisition_id: str) -> bytes:
            """Get raw waveform data for an acquisition."""
            try:
                waveform_data = await self.data_store.get_acquisition(acquisition_id)
                if waveform_data:
                    return json.dumps(waveform_data).encode('utf-8')
                else:
                    return json.dumps({
                        "error": f"Acquisition {acquisition_id} not found"
                    }).encode('utf-8')
            except Exception as e:
                return json.dumps({"error": str(e)}).encode('utf-8')
        
        @self.mcp.resource("latest://channel{channel}/waveform")
        async def get_latest_waveform(channel: int) -> bytes:
            """Get latest waveform data for a specific channel."""
            try:
                waveform_data = await self.data_store.get_latest_for_channel(channel)
                if waveform_data:
                    return json.dumps(waveform_data).encode('utf-8')
                else:
                    return json.dumps({
                        "error": f"No data available for channel {channel}"
                    }).encode('utf-8')
            except Exception as e:
                return json.dumps({"error": str(e)}).encode('utf-8')
    
    def _register_prompts(self):
        """Register MCP prompts for workflows."""
        
        @self.mcp.prompt
        async def adc_integration_workflow():
            """Workflow for integrating ADC hardware."""
            return """
# ADC Hardware Integration Workflow

## 1. Connect Your ADC Hardware
Configure your ADC to send data to this MCP server:

**HTTP Endpoint**: POST http://localhost:8080/adc/data
**WebSocket**: ws://localhost:8080/adc/stream

## 2. Data Format
Send data in this JSON format:
```json
{
    "device_id": "your_adc_device_name",
    "sample_rate": 44100.0,
    "timestamp": 1234567890.123,
    "channels": {
        "0": [0.1, 0.2, 0.3, ...],
        "1": [0.4, 0.5, 0.6, ...]
    },
    "metadata": {
        "voltage_range": 3.3,
        "coupling": "DC",
        "impedance": "1M"
    }
}
```

## 3. Use MCP Tools
Once data is flowing, use these MCP tools:
- `acquire_waveform()` - Wait for new data
- `measure_parameters()` - Analyze signals
- `analyze_spectrum()` - FFT analysis
- `decode_protocol()` - Protocol analysis

## 4. Access Data via Resources
- `acquisition://{id}/waveform` - Raw waveform data
- `latest://channel{n}/waveform` - Latest data per channel

This allows any ADC hardware to work with the MCP oscilloscope server!
            """
        
        @self.mcp.prompt
        async def microphone_test_workflow():
            """Workflow for testing with microphone as ADC."""
            return """
# Microphone ADC Testing Workflow

## Setup
1. Run the MCP server: `python -m oscilloscope_mcp.mcp_server`
2. Use the microphone client to send audio data as ADC input
3. Analyze real-time audio signals through the MCP interface

## Test Scenarios
- **Voice Analysis**: Speak and measure frequency content
- **Music Analysis**: Play music to test complex signals  
- **Tone Testing**: Use tone generator apps for calibration
- **Noise Floor**: Test in quiet environment

## Expected Results
- Real-time waveform data available via MCP tools
- FFT analysis showing audio frequency content
- Measurements tracking signal characteristics
- Protocol compliance for AI agent integration

This proves the MCP server can handle real hardware inputs!
            """
    
    async def initialize(self):
        """Initialize the MCP server components."""
        logger.info("Initializing MCP server components")
        
        try:
            # Initialize signal processing components
            await self.signal_processor.initialize()
            await self.measurement_analyzer.initialize()
            await self.protocol_decoder.initialize()
            
            logger.info("MCP server components initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize MCP server components", error=str(e))
            return False
    
    async def start(self):
        """Start the MCP server."""
        logger.info("Starting MCP server", host=self.host, port=self.port)
        
        try:
            # Initialize components
            if not await self.initialize():
                raise RuntimeError("Failed to initialize server components")
            
            # Create the ASGI app with additional routes
            app = self.mcp.create_app()
            
            # Add our custom routes
            for route in self.additional_routes:
                app.routes.append(route)
            
            self.is_running = True
            logger.info("MCP server started successfully", 
                       host=self.host, port=self.port,
                       websocket_enabled=self.config["enable_websocket"])
            
            # Run the server
            uvicorn_config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                log_level=self.config["log_level"].lower(),
                access_log=True
            )
            
            server = uvicorn.Server(uvicorn_config)
            await server.serve()
            
        except Exception as e:
            logger.error("Failed to start MCP server", error=str(e))
            raise
    
    async def stop(self):
        """Stop the MCP server."""
        logger.info("Stopping MCP server")
        
        try:
            self.is_running = False
            
            # Cleanup components
            if hasattr(self.signal_processor, 'cleanup'):
                await self.signal_processor.cleanup()
            
            logger.info("MCP server stopped successfully")
            
        except Exception as e:
            logger.error("Error stopping MCP server", error=str(e))
            raise


async def main():
    """Main entry point for the MCP server."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration from environment
    host = os.getenv("MCP_HOST", "localhost")
    port = int(os.getenv("MCP_PORT", "8080"))
    
    # Create and start server
    server = OscilloscopeMCPServer(host=host, port=port)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error("Server error", error=str(e))
        traceback.print_exc()
        sys.exit(1)
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())