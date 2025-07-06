"""
Prompt template manager for measurement workflows.
"""

from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)

class PromptManager:
    """Prompt template manager."""
    
    def __init__(self):
        self.templates = {
            "oscilloscope_basic_setup": """
# Basic Oscilloscope Setup Procedure

Follow this procedure to set up the oscilloscope for common measurements:

## 1. Configure Input Channels
- Set appropriate voltage range for your signal levels
- Choose coupling (AC/DC) based on signal characteristics
- Set input impedance (50Ω for high-frequency, 1MΩ for general use)

## 2. Set Timebase
- Configure sample rate to adequately sample your signal (>2x signal frequency)
- Set record length to capture the desired time window

## 3. Configure Trigger
- Select trigger source (usually the channel with your signal)
- Set trigger level to a stable point on your waveform
- Choose appropriate edge (rising/falling)

## 4. Acquire Data
- Start acquisition and verify stable triggering
- Adjust trigger level if needed for stable display

## 5. Make Measurements
- Use automated measurements for standard parameters
- Perform FFT analysis for frequency domain information

Use these MCP tools in sequence:
- configure_channels()
- set_timebase() 
- setup_trigger()
- acquire_waveform()
- measure_parameters()
            """,
            
            "signal_integrity_analysis": """
# Signal Integrity Analysis Workflow

Comprehensive procedure for analyzing signal integrity:

## 1. High-Resolution Acquisition
- Use maximum sample rate for your signal bandwidth
- Capture sufficient record length for statistical analysis
- Use AC coupling if analyzing signal transitions

## 2. Time Domain Measurements
- Rise time and fall time analysis
- Overshoot and undershoot measurements
- Signal amplitude and DC offset

## 3. Frequency Domain Analysis
- FFT analysis to identify harmonic content
- Check for spurious signals and noise
- Measure bandwidth and frequency response

## 4. Anomaly Detection
- Use statistical analysis to find glitches
- Look for timing violations and jitter
- Check for signal integrity violations

## 5. Eye Diagram Analysis (if applicable)
- For digital signals, generate eye diagrams
- Measure eye opening and timing margins
- Analyze jitter characteristics

Tools to use:
- acquire_waveform() with high sample rate
- measure_parameters() for timing measurements  
- analyze_spectrum() for frequency analysis
- detect_glitches() for anomaly detection
            """
        }
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize prompt manager."""
        self.is_initialized = True
        logger.info("Prompt manager initialized")
    
    async def get_prompt(self, prompt_name: str) -> str:
        """Get prompt template."""
        return self.templates.get(prompt_name, f"Prompt '{prompt_name}' not found")