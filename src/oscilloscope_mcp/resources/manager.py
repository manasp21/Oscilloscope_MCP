"""
Resource manager for waveform and analysis data.
"""

import time
from typing import Any, Dict, List, Optional
import structlog

logger = structlog.get_logger(__name__)

class ResourceManager:
    """Resource manager for data storage and retrieval."""
    
    def __init__(self):
        self.waveform_cache = {}
        self.spectrum_cache = {}
        self.measurement_cache = {}
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize resource manager."""
        self.is_initialized = True
        logger.info("Resource manager initialized")
    
    async def store_waveform(
        self,
        channels: List[int],
        data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Store waveform data."""
        acquisition_id = f"waveform_{int(time.time() * 1000)}"
        
        self.waveform_cache[acquisition_id] = {
            "channels": channels,
            "data": data,
            "metadata": metadata,
            "timestamp": time.time()
        }
        
        # Also store by channel for easy access
        for channel in channels:
            self.waveform_cache[f"channel_{channel}_latest"] = self.waveform_cache[acquisition_id]
        
        return acquisition_id
    
    async def store_spectrum(self, channel: int, data: Dict[str, Any]) -> str:
        """Store spectrum data."""
        spectrum_id = f"spectrum_{channel}_{int(time.time() * 1000)}"
        
        self.spectrum_cache[spectrum_id] = {
            "channel": channel,
            "data": data,
            "timestamp": time.time()
        }
        
        self.spectrum_cache[f"channel_{channel}_spectrum_latest"] = self.spectrum_cache[spectrum_id]
        
        return spectrum_id
    
    async def get_latest_waveform(self, channel: int) -> Optional[Dict[str, Any]]:
        """Get latest waveform data for channel."""
        key = f"channel_{channel}_latest"
        if key in self.waveform_cache:
            return self.waveform_cache[key]["data"]
        return None
    
    async def get_latest_spectrum(self, channel: int) -> Optional[Dict[str, Any]]:
        """Get latest spectrum data for channel."""
        key = f"channel_{channel}_spectrum_latest"
        if key in self.spectrum_cache:
            return self.spectrum_cache[key]["data"]
        return None