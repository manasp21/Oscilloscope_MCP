"""
Main MCP server implementation for oscilloscope and function generator.

This module provides the core MCP server that exposes oscilloscope and function
generator capabilities through the Model Context Protocol.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import structlog
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .hardware.interface import HardwareInterface
from .signal_processing.engine import SignalProcessor
from .measurements.analyzer import MeasurementAnalyzer
from .protocols.decoder import ProtocolDecoder
from .resources.manager import ResourceManager
from .prompts.templates import PromptManager


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


class OscilloscopeMCPServer:
    """Main MCP server class for oscilloscope and function generator."""
    
    def __init__(self):
        """Initialize the MCP server with all components."""
        self.mcp = FastMCP("oscilloscope-function-generator")
        
        # Initialize core components
        self.hardware = HardwareInterface()
        self.signal_processor = SignalProcessor()
        self.measurement_analyzer = MeasurementAnalyzer()
        self.protocol_decoder = ProtocolDecoder()
        self.resource_manager = ResourceManager()
        self.prompt_manager = PromptManager()
        
        # Server state
        self.is_running = False
        self.acquisition_active = False
        self.generation_active = False
        
        # Configuration
        self.config = self._load_config()
        
        # Register all MCP endpoints
        self._register_oscilloscope_tools()
        self._register_function_generator_tools()
        self._register_analysis_tools()
        self._register_resources()
        self._register_prompts()
        
        logger.info("MCP server initialized", config=self.config)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load server configuration from environment variables."""
        return {
            "hardware_interface": os.getenv("HARDWARE_INTERFACE", "simulation"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "sample_rate": float(os.getenv("SAMPLE_RATE", "100e6")),
            "channels": int(os.getenv("CHANNELS", "4")),
            "buffer_size": int(os.getenv("BUFFER_SIZE", "1048576")),
            "timeout": float(os.getenv("TIMEOUT", "5.0")),
        }
    
    def _register_oscilloscope_tools(self):
        """Register oscilloscope-related MCP tools."""
        
        @self.mcp.tool
        async def configure_channels(
            channels: List[int],
            voltage_range: float,
            coupling: str = "DC",
            impedance: str = "1M"
        ) -> Dict[str, Any]:
            """Configure oscilloscope input channels."""
            logger.info("Configuring channels", channels=channels, voltage_range=voltage_range)
            
            try:
                result = await self.hardware.configure_channels(
                    channels=channels,
                    voltage_range=voltage_range,
                    coupling=coupling,
                    impedance=impedance
                )
                
                logger.info("Channels configured successfully", result=result)
                return {
                    "status": "success",
                    "channels_configured": channels,
                    "voltage_range": voltage_range,
                    "coupling": coupling,
                    "impedance": impedance,
                    "timestamp": result.get("timestamp")
                }
                
            except Exception as e:
                logger.error("Failed to configure channels", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "channels": channels
                }
        
        @self.mcp.tool
        async def set_timebase(
            sample_rate: float,
            record_length: int
        ) -> Dict[str, Any]:
            """Configure acquisition timebase."""
            logger.info("Setting timebase", sample_rate=sample_rate, record_length=record_length)
            
            try:
                result = await self.hardware.set_timebase(
                    sample_rate=sample_rate,
                    record_length=record_length
                )
                
                logger.info("Timebase configured successfully", result=result)
                return {
                    "status": "success",
                    "sample_rate": sample_rate,
                    "record_length": record_length,
                    "actual_sample_rate": result.get("actual_sample_rate"),
                    "memory_depth": result.get("memory_depth")
                }
                
            except Exception as e:
                logger.error("Failed to set timebase", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "requested_sample_rate": sample_rate,
                    "requested_record_length": record_length
                }
        
        @self.mcp.tool
        async def setup_trigger(
            source: str,
            trigger_type: str,
            level: float,
            edge: str = "rising",
            holdoff: float = 0.0
        ) -> Dict[str, Any]:
            """Configure trigger conditions."""
            logger.info("Setting up trigger", source=source, type=trigger_type, level=level)
            
            try:
                result = await self.hardware.setup_trigger(
                    source=source,
                    trigger_type=trigger_type,
                    level=level,
                    edge=edge,
                    holdoff=holdoff
                )
                
                logger.info("Trigger configured successfully", result=result)
                return {
                    "status": "success",
                    "source": source,
                    "type": trigger_type,
                    "level": level,
                    "edge": edge,
                    "holdoff": holdoff,
                    "trigger_id": result.get("trigger_id")
                }
                
            except Exception as e:
                logger.error("Failed to setup trigger", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "source": source,
                    "type": trigger_type
                }
        
        @self.mcp.tool
        async def acquire_waveform(
            channels: List[int],
            timeout: float = 5.0
        ) -> Dict[str, Any]:
            """Acquire waveform data from specified channels."""
            logger.info("Acquiring waveform", channels=channels, timeout=timeout)
            
            try:
                self.acquisition_active = True
                
                # Acquire raw data from hardware
                raw_data = await self.hardware.acquire_waveform(
                    channels=channels,
                    timeout=timeout
                )
                
                # Process the acquired data
                processed_data = await self.signal_processor.process_waveform(raw_data)
                
                # Store in resource manager for later access
                acquisition_id = await self.resource_manager.store_waveform(
                    channels=channels,
                    data=processed_data,
                    metadata=raw_data.get("metadata", {})
                )
                
                logger.info("Waveform acquired successfully", 
                           channels=channels, 
                           acquisition_id=acquisition_id,
                           samples=len(processed_data.get("time", [])))
                
                return {
                    "status": "success",
                    "channels": channels,
                    "acquisition_id": acquisition_id,
                    "samples": len(processed_data.get("time", [])),
                    "sample_rate": processed_data.get("sample_rate"),
                    "timestamp": processed_data.get("timestamp"),
                    "waveform_uri": f"waveform://channel{channels[0]}/buffer"
                }
                
            except Exception as e:
                logger.error("Failed to acquire waveform", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "channels": channels
                }
            finally:
                self.acquisition_active = False
        
        @self.mcp.tool
        async def measure_parameters(
            channel: int,
            measurements: List[str],
            statistics: bool = False
        ) -> Dict[str, Any]:
            """Perform automated measurements on waveform data."""
            logger.info("Measuring parameters", channel=channel, measurements=measurements)
            
            try:
                # Get the latest waveform data for the channel
                waveform_data = await self.resource_manager.get_latest_waveform(channel)
                
                if not waveform_data:
                    return {
                        "status": "error",
                        "error": "No waveform data available for channel",
                        "channel": channel
                    }
                
                # Perform measurements
                results = await self.measurement_analyzer.measure_parameters(
                    waveform_data=waveform_data,
                    measurements=measurements,
                    statistics=statistics
                )
                
                logger.info("Measurements completed", 
                           channel=channel, 
                           measurements=measurements,
                           results=results)
                
                return {
                    "status": "success",
                    "channel": channel,
                    "measurements": measurements,
                    "results": results,
                    "timestamp": results.get("timestamp")
                }
                
            except Exception as e:
                logger.error("Failed to measure parameters", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "channel": channel,
                    "measurements": measurements
                }
        
        @self.mcp.tool
        async def analyze_spectrum(
            channel: int,
            window: str = "hamming",
            resolution: int = 1024
        ) -> Dict[str, Any]:
            """Perform FFT analysis on waveform data."""
            logger.info("Analyzing spectrum", channel=channel, window=window, resolution=resolution)
            
            try:
                # Get the latest waveform data for the channel
                waveform_data = await self.resource_manager.get_latest_waveform(channel)
                
                if not waveform_data:
                    return {
                        "status": "error",
                        "error": "No waveform data available for channel",
                        "channel": channel
                    }
                
                # Perform FFT analysis
                spectrum_data = await self.signal_processor.analyze_spectrum(
                    waveform_data=waveform_data,
                    window=window,
                    resolution=resolution
                )
                
                # Store spectrum data in resource manager
                spectrum_id = await self.resource_manager.store_spectrum(
                    channel=channel,
                    data=spectrum_data
                )
                
                logger.info("Spectrum analysis completed", 
                           channel=channel, 
                           spectrum_id=spectrum_id,
                           frequency_bins=len(spectrum_data.get("frequency", [])))
                
                return {
                    "status": "success",
                    "channel": channel,
                    "spectrum_id": spectrum_id,
                    "window": window,
                    "resolution": resolution,
                    "frequency_bins": len(spectrum_data.get("frequency", [])),
                    "spectrum_uri": f"spectrum://channel{channel}/fft"
                }
                
            except Exception as e:
                logger.error("Failed to analyze spectrum", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "channel": channel
                }
    
    def _register_function_generator_tools(self):
        """Register function generator-related MCP tools."""
        
        @self.mcp.tool
        async def generate_standard_waveform(
            channel: int,
            waveform: str,
            frequency: float,
            amplitude: float,
            offset: float = 0.0,
            phase: float = 0.0
        ) -> Dict[str, Any]:
            """Generate standard waveforms."""
            logger.info("Generating standard waveform", 
                       channel=channel, 
                       waveform=waveform, 
                       frequency=frequency)
            
            try:
                self.generation_active = True
                
                result = await self.hardware.generate_standard_waveform(
                    channel=channel,
                    waveform=waveform,
                    frequency=frequency,
                    amplitude=amplitude,
                    offset=offset,
                    phase=phase
                )
                
                logger.info("Standard waveform generated successfully", result=result)
                return {
                    "status": "success",
                    "channel": channel,
                    "waveform": waveform,
                    "frequency": frequency,
                    "amplitude": amplitude,
                    "offset": offset,
                    "phase": phase,
                    "generation_id": result.get("generation_id")
                }
                
            except Exception as e:
                logger.error("Failed to generate standard waveform", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "channel": channel,
                    "waveform": waveform
                }
        
        @self.mcp.tool
        async def generate_arbitrary_waveform(
            channel: int,
            samples: List[float],
            sample_rate: float,
            amplitude: float = 1.0,
            offset: float = 0.0
        ) -> Dict[str, Any]:
            """Generate arbitrary waveform from sample data."""
            logger.info("Generating arbitrary waveform", 
                       channel=channel, 
                       samples_count=len(samples),
                       sample_rate=sample_rate)
            
            try:
                self.generation_active = True
                
                result = await self.hardware.generate_arbitrary_waveform(
                    channel=channel,
                    samples=samples,
                    sample_rate=sample_rate,
                    amplitude=amplitude,
                    offset=offset
                )
                
                logger.info("Arbitrary waveform generated successfully", result=result)
                return {
                    "status": "success",
                    "channel": channel,
                    "samples_count": len(samples),
                    "sample_rate": sample_rate,
                    "amplitude": amplitude,
                    "offset": offset,
                    "generation_id": result.get("generation_id")
                }
                
            except Exception as e:
                logger.error("Failed to generate arbitrary waveform", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "channel": channel,
                    "samples_count": len(samples)
                }
    
    def _register_analysis_tools(self):
        """Register advanced analysis tools."""
        
        @self.mcp.tool
        async def decode_protocol(
            channels: List[int],
            protocol: str,
            settings: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Decode digital communication protocols."""
            logger.info("Decoding protocol", channels=channels, protocol=protocol)
            
            try:
                # Get waveform data for all specified channels
                waveform_data = {}
                for channel in channels:
                    data = await self.resource_manager.get_latest_waveform(channel)
                    if data:
                        waveform_data[channel] = data
                
                if not waveform_data:
                    return {
                        "status": "error",
                        "error": "No waveform data available for specified channels",
                        "channels": channels
                    }
                
                # Decode the protocol
                decoded_data = await self.protocol_decoder.decode_protocol(
                    waveform_data=waveform_data,
                    protocol=protocol,
                    settings=settings
                )
                
                logger.info("Protocol decoded successfully", 
                           protocol=protocol, 
                           packets=len(decoded_data.get("packets", [])))
                
                return {
                    "status": "success",
                    "channels": channels,
                    "protocol": protocol,
                    "packets": decoded_data.get("packets", []),
                    "errors": decoded_data.get("errors", []),
                    "statistics": decoded_data.get("statistics", {}),
                    "timestamp": decoded_data.get("timestamp")
                }
                
            except Exception as e:
                logger.error("Failed to decode protocol", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "channels": channels,
                    "protocol": protocol
                }
    
    def _register_resources(self):
        """Register MCP resources for data access."""
        
        @self.mcp.resource("waveform://channel{channel}/buffer")
        async def get_waveform_data(channel: int) -> bytes:
            """Get waveform data for a specific channel."""
            try:
                data = await self.resource_manager.get_latest_waveform(channel)
                if data:
                    return json.dumps(data).encode('utf-8')
                else:
                    return b'{"error": "No waveform data available"}'
            except Exception as e:
                return json.dumps({"error": str(e)}).encode('utf-8')
        
        @self.mcp.resource("spectrum://channel{channel}/fft")
        async def get_spectrum_data(channel: int) -> bytes:
            """Get spectrum data for a specific channel."""
            try:
                data = await self.resource_manager.get_latest_spectrum(channel)
                if data:
                    return json.dumps(data).encode('utf-8')
                else:
                    return b'{"error": "No spectrum data available"}'
            except Exception as e:
                return json.dumps({"error": str(e)}).encode('utf-8')
    
    def _register_prompts(self):
        """Register MCP prompts for measurement workflows."""
        
        @self.mcp.prompt
        async def oscilloscope_basic_setup():
            """Basic oscilloscope setup procedure."""
            return await self.prompt_manager.get_prompt("oscilloscope_basic_setup")
        
        @self.mcp.prompt
        async def signal_integrity_analysis():
            """Signal integrity analysis workflow."""
            return await self.prompt_manager.get_prompt("signal_integrity_analysis")
    
    async def start(self):
        """Start the MCP server."""
        logger.info("Starting MCP server")
        
        try:
            # Initialize hardware interface
            await self.hardware.initialize(self.config)
            
            # Initialize signal processor
            await self.signal_processor.initialize()
            
            # Initialize other components
            await self.measurement_analyzer.initialize()
            await self.protocol_decoder.initialize()
            
            self.is_running = True
            logger.info("MCP server started successfully")
            
            # Run the MCP server
            await self.mcp.run()
            
        except Exception as e:
            logger.error("Failed to start MCP server", error=str(e))
            raise
    
    async def stop(self):
        """Stop the MCP server."""
        logger.info("Stopping MCP server")
        
        try:
            self.is_running = False
            
            # Stop any active acquisitions or generations
            if self.acquisition_active:
                await self.hardware.stop_acquisition()
                
            if self.generation_active:
                await self.hardware.stop_generation()
            
            # Cleanup hardware interface
            await self.hardware.cleanup()
            
            logger.info("MCP server stopped successfully")
            
        except Exception as e:
            logger.error("Error stopping MCP server", error=str(e))
            raise


# Main entry point
async def main():
    """Main entry point for the MCP server."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start server
    server = OscilloscopeMCPServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error("Server error", error=str(e))
        sys.exit(1)
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())