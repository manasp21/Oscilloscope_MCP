"""
FFT analysis module for signal processing.
"""

import numpy as np
from scipy import signal
from typing import Any, Dict, List, Optional
import structlog

logger = structlog.get_logger(__name__)

class FFTAnalyzer:
    """FFT analysis functionality."""
    
    def __init__(self):
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize FFT analyzer."""
        self.is_initialized = True
        logger.info("FFT analyzer initialized")
    
    async def compute_fft(
        self,
        signal_data: np.ndarray,
        sample_rate: float,
        window: str = "hamming",
        nfft: int = 1024,
        overlap: float = 0.5
    ) -> Dict[str, Any]:
        """Compute FFT with windowing."""
        
        # Apply window
        if window == "hamming":
            windowed = signal_data * np.hamming(len(signal_data))
        elif window == "blackman":
            windowed = signal_data * np.blackman(len(signal_data))
        else:
            windowed = signal_data
        
        # Compute FFT
        fft_result = np.fft.fft(windowed, n=nfft)
        frequency = np.fft.fftfreq(nfft, 1/sample_rate)
        
        # Only keep positive frequencies
        positive_freqs = frequency[:nfft//2]
        positive_fft = fft_result[:nfft//2]
        
        magnitude = np.abs(positive_fft)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        phase = np.angle(positive_fft, deg=True)
        
        return {
            "frequency": positive_freqs.tolist(),
            "magnitude": magnitude.tolist(),
            "magnitude_db": magnitude_db.tolist(),
            "phase_deg": phase.tolist(),
            "complex": positive_fft.tolist()
        }