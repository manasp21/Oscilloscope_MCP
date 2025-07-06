"""
Digital filtering module.
"""

import numpy as np
from scipy import signal
from typing import List, Union
import structlog

logger = structlog.get_logger(__name__)

class DigitalFilter:
    """Digital filtering functionality."""
    
    def __init__(self):
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize digital filter."""
        self.is_initialized = True
        logger.info("Digital filter initialized")
    
    async def apply_lowpass_filter(
        self, 
        data: np.ndarray, 
        sample_rate: float, 
        cutoff: float, 
        order: int = 4
    ) -> np.ndarray:
        """Apply lowpass filter."""
        sos = signal.butter(order, cutoff / (sample_rate / 2), 'low', output='sos')
        return signal.sosfilt(sos, data)
    
    async def apply_highpass_filter(
        self, 
        data: np.ndarray, 
        sample_rate: float, 
        cutoff: float, 
        order: int = 4
    ) -> np.ndarray:
        """Apply highpass filter."""
        sos = signal.butter(order, cutoff / (sample_rate / 2), 'high', output='sos')
        return signal.sosfilt(sos, data)
    
    async def apply_bandpass_filter(
        self, 
        data: np.ndarray, 
        sample_rate: float, 
        cutoff: List[float], 
        order: int = 4
    ) -> np.ndarray:
        """Apply bandpass filter."""
        sos = signal.butter(order, [cutoff[0] / (sample_rate / 2), cutoff[1] / (sample_rate / 2)], 'band', output='sos')
        return signal.sosfilt(sos, data)
    
    async def apply_bandstop_filter(
        self, 
        data: np.ndarray, 
        sample_rate: float, 
        cutoff: List[float], 
        order: int = 4
    ) -> np.ndarray:
        """Apply bandstop filter."""
        sos = signal.butter(order, [cutoff[0] / (sample_rate / 2), cutoff[1] / (sample_rate / 2)], 'bandstop', output='sos')
        return signal.sosfilt(sos, data)