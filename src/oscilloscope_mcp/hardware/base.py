"""
Base classes for hardware backend abstraction.

This module defines the abstract base classes that all hardware backends
must implement to provide a consistent interface for the MCP server.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np


class OscilloscopeBackend(ABC):
    """Abstract base class for oscilloscope hardware backends."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the oscilloscope hardware."""
        pass
    
    @abstractmethod
    async def configure_channels(
        self,
        channels: List[int],
        voltage_range: float,
        coupling: str,
        impedance: str
    ) -> Dict[str, Any]:
        """Configure input channels."""
        pass
    
    @abstractmethod
    async def set_timebase(
        self,
        sample_rate: float,
        record_length: int
    ) -> Dict[str, Any]:
        """Configure acquisition timebase."""
        pass
    
    @abstractmethod
    async def setup_trigger(
        self,
        source: str,
        trigger_type: str,
        level: float,
        edge: str,
        holdoff: float
    ) -> Dict[str, Any]:
        """Configure trigger conditions."""
        pass
    
    @abstractmethod
    async def acquire_waveform(
        self,
        channels: List[int],
        timeout: float
    ) -> Dict[str, Any]:
        """Acquire waveform data from specified channels."""
        pass
    
    @abstractmethod
    async def stop_acquisition(self) -> None:
        """Stop any ongoing acquisition."""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current oscilloscope status."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class FunctionGeneratorBackend(ABC):
    """Abstract base class for function generator hardware backends."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the function generator hardware."""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def generate_arbitrary_waveform(
        self,
        channel: int,
        samples: List[float],
        sample_rate: float,
        amplitude: float,
        offset: float
    ) -> Dict[str, Any]:
        """Generate arbitrary waveform from sample data."""
        pass
    
    @abstractmethod
    async def configure_modulation(
        self,
        channel: int,
        mod_type: str,
        carrier_freq: float,
        mod_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure modulation settings."""
        pass
    
    @abstractmethod
    async def setup_sweep(
        self,
        channel: int,
        start_freq: float,
        stop_freq: float,
        sweep_time: float,
        sweep_type: str
    ) -> Dict[str, Any]:
        """Configure frequency sweep."""
        pass
    
    @abstractmethod
    async def stop_generation(self) -> None:
        """Stop signal generation."""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current function generator status."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class HardwareBackend(ABC):
    """Combined hardware backend interface."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the hardware."""
        pass
    
    @abstractmethod
    async def get_oscilloscope(self) -> OscilloscopeBackend:
        """Get oscilloscope backend."""
        pass
    
    @abstractmethod
    async def get_function_generator(self) -> FunctionGeneratorBackend:
        """Get function generator backend."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup all resources."""
        pass


class InstrumentCapabilities:
    """Class to define instrument capabilities and specifications."""
    
    def __init__(self):
        self.oscilloscope_specs = {
            "max_sample_rate": 1e9,  # 1 GS/s
            "max_bandwidth": 500e6,  # 500 MHz
            "max_channels": 4,
            "max_memory_depth": 1e9,  # 1 GSample
            "min_voltage_range": 0.001,  # 1 mV/div
            "max_voltage_range": 100.0,  # 100 V/div
            "supported_coupling": ["DC", "AC", "GND"],
            "supported_impedance": ["50", "1M"],
            "trigger_types": ["edge", "pulse", "pattern", "protocol"],
            "measurement_types": [
                "amplitude", "rms", "peak_to_peak", "frequency", "period",
                "rise_time", "fall_time", "duty_cycle", "overshoot",
                "undershoot", "phase", "delay"
            ]
        }
        
        self.function_generator_specs = {
            "max_frequency": 500e6,  # 500 MHz
            "min_frequency": 1e-6,   # 1 µHz
            "max_amplitude": 10.0,   # 10 V
            "min_amplitude": 0.001,  # 1 mV
            "max_offset": 5.0,       # ±5 V
            "max_channels": 2,
            "supported_waveforms": [
                "sine", "square", "triangle", "sawtooth", "pulse",
                "noise", "dc", "arbitrary"
            ],
            "modulation_types": ["AM", "FM", "PM", "FSK", "PSK", "QAM"],
            "sweep_types": ["linear", "logarithmic"],
            "max_arbitrary_length": 1e6,  # 1M samples
            "frequency_resolution": 1e-6,  # 1 µHz
            "amplitude_accuracy": 0.01,     # 1%
            "phase_accuracy": 0.1          # 0.1 degrees
        }
    
    def validate_oscilloscope_config(self, config: Dict[str, Any]) -> bool:
        """Validate oscilloscope configuration against capabilities."""
        if config.get("sample_rate", 0) > self.oscilloscope_specs["max_sample_rate"]:
            return False
        if config.get("channels", 0) > self.oscilloscope_specs["max_channels"]:
            return False
        if config.get("voltage_range", 0) > self.oscilloscope_specs["max_voltage_range"]:
            return False
        return True
    
    def validate_function_generator_config(self, config: Dict[str, Any]) -> bool:
        """Validate function generator configuration against capabilities."""
        if config.get("frequency", 0) > self.function_generator_specs["max_frequency"]:
            return False
        if config.get("amplitude", 0) > self.function_generator_specs["max_amplitude"]:
            return False
        if config.get("channel", 0) > self.function_generator_specs["max_channels"]:
            return False
        return True