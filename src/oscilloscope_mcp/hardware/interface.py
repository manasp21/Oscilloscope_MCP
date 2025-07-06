"""
Hardware interface manager for oscilloscope and function generator.

This module provides a unified interface that manages different hardware backends
and routes requests to the appropriate implementation (simulation or real hardware).
"""

import os
from typing import Any, Dict, List, Optional, Union
import structlog

from .base import HardwareBackend, OscilloscopeBackend, FunctionGeneratorBackend
from .simulation import SimulatedHardware


logger = structlog.get_logger(__name__)


class HardwareInterface:
    """Unified hardware interface manager."""
    
    def __init__(self):
        self.backend: Optional[HardwareBackend] = None
        self.oscilloscope: Optional[OscilloscopeBackend] = None
        self.function_generator: Optional[FunctionGeneratorBackend] = None
        self.config = {}
        self.is_initialized = False
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the hardware interface with specified backend."""
        self.config = config
        interface_type = config.get("hardware_interface", "simulation")
        
        logger.info("Initializing hardware interface", interface_type=interface_type)
        
        try:
            # Create appropriate backend
            if interface_type == "simulation":
                self.backend = SimulatedHardware()
            elif interface_type == "usb":
                self.backend = await self._create_usb_backend()
            elif interface_type == "ethernet":
                self.backend = await self._create_ethernet_backend()
            elif interface_type == "pcie":
                self.backend = await self._create_pcie_backend()
            else:
                raise ValueError(f"Unsupported interface type: {interface_type}")
            
            # Initialize the backend
            await self.backend.initialize(config)
            
            # Get component interfaces
            self.oscilloscope = await self.backend.get_oscilloscope()
            self.function_generator = await self.backend.get_function_generator()
            
            self.is_initialized = True
            logger.info("Hardware interface initialized successfully", interface_type=interface_type)
            
        except Exception as e:
            logger.error("Failed to initialize hardware interface", error=str(e))
            raise
    
    async def _create_usb_backend(self) -> HardwareBackend:
        """Create USB hardware backend."""
        try:
            from .usb_backend import USBHardware
            return USBHardware()
        except ImportError:
            logger.warning("USB backend not available, falling back to simulation")
            return SimulatedHardware()
    
    async def _create_ethernet_backend(self) -> HardwareBackend:
        """Create Ethernet hardware backend."""
        try:
            from .ethernet_backend import EthernetHardware
            return EthernetHardware()
        except ImportError:
            logger.warning("Ethernet backend not available, falling back to simulation")
            return SimulatedHardware()
    
    async def _create_pcie_backend(self) -> HardwareBackend:
        """Create PCIe hardware backend."""
        try:
            from .pcie_backend import PCIeHardware
            return PCIeHardware()
        except ImportError:
            logger.warning("PCIe backend not available, falling back to simulation")
            return SimulatedHardware()
    
    # Oscilloscope interface methods
    async def configure_channels(
        self,
        channels: List[int],
        voltage_range: float,
        coupling: str,
        impedance: str
    ) -> Dict[str, Any]:
        """Configure oscilloscope input channels."""
        if not self.is_initialized or not self.oscilloscope:
            raise RuntimeError("Hardware interface not initialized")
        
        return await self.oscilloscope.configure_channels(
            channels=channels,
            voltage_range=voltage_range,
            coupling=coupling,
            impedance=impedance
        )
    
    async def set_timebase(
        self,
        sample_rate: float,
        record_length: int
    ) -> Dict[str, Any]:
        """Configure acquisition timebase."""
        if not self.is_initialized or not self.oscilloscope:
            raise RuntimeError("Hardware interface not initialized")
        
        return await self.oscilloscope.set_timebase(
            sample_rate=sample_rate,
            record_length=record_length
        )
    
    async def setup_trigger(
        self,
        source: str,
        trigger_type: str,
        level: float,
        edge: str,
        holdoff: float
    ) -> Dict[str, Any]:
        """Configure trigger conditions."""
        if not self.is_initialized or not self.oscilloscope:
            raise RuntimeError("Hardware interface not initialized")
        
        return await self.oscilloscope.setup_trigger(
            source=source,
            trigger_type=trigger_type,
            level=level,
            edge=edge,
            holdoff=holdoff
        )
    
    async def acquire_waveform(
        self,
        channels: List[int],
        timeout: float
    ) -> Dict[str, Any]:
        """Acquire waveform data from specified channels."""
        if not self.is_initialized or not self.oscilloscope:
            raise RuntimeError("Hardware interface not initialized")
        
        return await self.oscilloscope.acquire_waveform(
            channels=channels,
            timeout=timeout
        )
    
    async def stop_acquisition(self) -> None:
        """Stop any ongoing acquisition."""
        if self.oscilloscope:
            await self.oscilloscope.stop_acquisition()
    
    async def get_oscilloscope_status(self) -> Dict[str, Any]:
        """Get current oscilloscope status."""
        if not self.is_initialized or not self.oscilloscope:
            raise RuntimeError("Hardware interface not initialized")
        
        return await self.oscilloscope.get_status()
    
    # Function generator interface methods
    async def generate_standard_waveform(
        self,
        channel: int,
        waveform: str,
        frequency: float,
        amplitude: float,
        offset: float,
        phase: float
    ) -> Dict[str, Any]:
        """Generate standard waveforms."""
        if not self.is_initialized or not self.function_generator:
            raise RuntimeError("Hardware interface not initialized")
        
        return await self.function_generator.generate_standard_waveform(
            channel=channel,
            waveform=waveform,
            frequency=frequency,
            amplitude=amplitude,
            offset=offset,
            phase=phase
        )
    
    async def generate_arbitrary_waveform(
        self,
        channel: int,
        samples: List[float],
        sample_rate: float,
        amplitude: float,
        offset: float
    ) -> Dict[str, Any]:
        """Generate arbitrary waveform from sample data."""
        if not self.is_initialized or not self.function_generator:
            raise RuntimeError("Hardware interface not initialized")
        
        return await self.function_generator.generate_arbitrary_waveform(
            channel=channel,
            samples=samples,
            sample_rate=sample_rate,
            amplitude=amplitude,
            offset=offset
        )
    
    async def configure_modulation(
        self,
        channel: int,
        mod_type: str,
        carrier_freq: float,
        mod_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure modulation settings."""
        if not self.is_initialized or not self.function_generator:
            raise RuntimeError("Hardware interface not initialized")
        
        return await self.function_generator.configure_modulation(
            channel=channel,
            mod_type=mod_type,
            carrier_freq=carrier_freq,
            mod_params=mod_params
        )
    
    async def setup_sweep(
        self,
        channel: int,
        start_freq: float,
        stop_freq: float,
        sweep_time: float,
        sweep_type: str = "linear"
    ) -> Dict[str, Any]:
        """Configure frequency sweep."""
        if not self.is_initialized or not self.function_generator:
            raise RuntimeError("Hardware interface not initialized")
        
        return await self.function_generator.setup_sweep(
            channel=channel,
            start_freq=start_freq,
            stop_freq=stop_freq,
            sweep_time=sweep_time,
            sweep_type=sweep_type
        )
    
    async def stop_generation(self) -> None:
        """Stop signal generation."""
        if self.function_generator:
            await self.function_generator.stop_generation()
    
    async def get_function_generator_status(self) -> Dict[str, Any]:
        """Get current function generator status."""
        if not self.is_initialized or not self.function_generator:
            raise RuntimeError("Hardware interface not initialized")
        
        return await self.function_generator.get_status()
    
    # General interface methods
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        if not self.is_initialized:
            return {
                "initialized": False,
                "backend_type": None,
                "oscilloscope_status": None,
                "function_generator_status": None
            }
        
        osc_status = await self.get_oscilloscope_status() if self.oscilloscope else None
        fg_status = await self.get_function_generator_status() if self.function_generator else None
        
        return {
            "initialized": self.is_initialized,
            "backend_type": self.config.get("hardware_interface", "unknown"),
            "oscilloscope_status": osc_status,
            "function_generator_status": fg_status,
            "config": self.config
        }
    
    async def perform_self_test(self) -> Dict[str, Any]:
        """Perform system self-test."""
        logger.info("Starting hardware self-test")
        
        test_results = {
            "overall_status": "pass",
            "tests": {},
            "timestamp": None
        }
        
        try:
            # Test oscilloscope functionality
            if self.oscilloscope:
                logger.info("Testing oscilloscope functionality")
                
                # Test channel configuration
                test_channels = [0]
                try:
                    await self.configure_channels(
                        channels=test_channels,
                        voltage_range=1.0,
                        coupling="DC",
                        impedance="1M"
                    )
                    test_results["tests"]["oscilloscope_channel_config"] = "pass"
                except Exception as e:
                    test_results["tests"]["oscilloscope_channel_config"] = f"fail: {str(e)}"
                    test_results["overall_status"] = "fail"
                
                # Test timebase configuration
                try:
                    await self.set_timebase(sample_rate=1e6, record_length=1000)
                    test_results["tests"]["oscilloscope_timebase"] = "pass"
                except Exception as e:
                    test_results["tests"]["oscilloscope_timebase"] = f"fail: {str(e)}"
                    test_results["overall_status"] = "fail"
                
                # Test acquisition
                try:
                    await self.acquire_waveform(channels=test_channels, timeout=1.0)
                    test_results["tests"]["oscilloscope_acquisition"] = "pass"
                except Exception as e:
                    test_results["tests"]["oscilloscope_acquisition"] = f"fail: {str(e)}"
                    test_results["overall_status"] = "fail"
            
            # Test function generator functionality
            if self.function_generator:
                logger.info("Testing function generator functionality")
                
                # Test waveform generation
                try:
                    await self.generate_standard_waveform(
                        channel=0,
                        waveform="sine",
                        frequency=1000.0,
                        amplitude=1.0,
                        offset=0.0,
                        phase=0.0
                    )
                    test_results["tests"]["function_generator_waveform"] = "pass"
                except Exception as e:
                    test_results["tests"]["function_generator_waveform"] = f"fail: {str(e)}"
                    test_results["overall_status"] = "fail"
                
                # Test arbitrary waveform
                try:
                    test_samples = [0.0, 1.0, 0.0, -1.0] * 10
                    await self.generate_arbitrary_waveform(
                        channel=0,
                        samples=test_samples,
                        sample_rate=1000.0,
                        amplitude=1.0,
                        offset=0.0
                    )
                    test_results["tests"]["function_generator_arbitrary"] = "pass"
                except Exception as e:
                    test_results["tests"]["function_generator_arbitrary"] = f"fail: {str(e)}"
                    test_results["overall_status"] = "fail"
            
            import time
            test_results["timestamp"] = time.time()
            
            logger.info("Hardware self-test completed", 
                       overall_status=test_results["overall_status"],
                       tests=test_results["tests"])
            
        except Exception as e:
            logger.error("Hardware self-test failed", error=str(e))
            test_results["overall_status"] = "error"
            test_results["error"] = str(e)
        
        return test_results
    
    async def cleanup(self) -> None:
        """Cleanup hardware interface."""
        logger.info("Cleaning up hardware interface")
        
        try:
            # Stop any active operations
            await self.stop_acquisition()
            await self.stop_generation()
            
            # Cleanup backend
            if self.backend:
                await self.backend.cleanup()
            
            self.is_initialized = False
            self.oscilloscope = None
            self.function_generator = None
            self.backend = None
            
            logger.info("Hardware interface cleaned up successfully")
            
        except Exception as e:
            logger.error("Error during hardware cleanup", error=str(e))
            raise