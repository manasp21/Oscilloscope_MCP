"""
Hardware interface module for oscilloscope and function generator.

This module provides the hardware abstraction layer that supports both
simulation and real hardware interfaces.
"""

from .interface import HardwareInterface
from .simulation import SimulatedHardware
from .base import OscilloscopeBackend, FunctionGeneratorBackend

__all__ = [
    "HardwareInterface",
    "SimulatedHardware", 
    "OscilloscopeBackend",
    "FunctionGeneratorBackend",
]