"""
Pytest configuration and shared fixtures for the oscilloscope MCP test suite.
"""

import pytest
import asyncio
import sys
import os
from typing import Dict, Any

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oscilloscope_mcp.hardware.simulation import SimulatedHardware
from oscilloscope_mcp.mcp_server import OscilloscopeMCPServer


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def simulated_hardware():
    """Provide initialized simulated hardware for testing."""
    hardware = SimulatedHardware()
    
    config = {
        "hardware_interface": "simulation",
        "sample_rate": 1e6,
        "channels": 4,
        "buffer_size": 1000,
        "timeout": 5.0,
    }
    
    await hardware.initialize(config)
    
    yield hardware
    
    await hardware.cleanup()


@pytest.fixture
async def mcp_server():
    """Provide initialized MCP server for testing."""
    server = OscilloscopeMCPServer()
    server.hardware = SimulatedHardware()
    
    config = {
        "hardware_interface": "simulation",
        "sample_rate": 1e6,
        "channels": 4,
        "buffer_size": 1000,
    }
    
    await server.hardware.initialize(config)
    
    yield server
    
    await server.hardware.cleanup()


@pytest.fixture
def sample_waveform_data():
    """Provide sample waveform data for testing."""
    import numpy as np
    
    # Generate test waveform: 1kHz sine wave + 3kHz harmonic
    sample_rate = 10000
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    fundamental = np.sin(2 * np.pi * 1000 * t)
    harmonic = 0.3 * np.sin(2 * np.pi * 3000 * t)
    noise = 0.05 * np.random.randn(len(t))
    
    signal = fundamental + harmonic + noise
    
    return {
        "time": t.tolist(),
        "signal": signal.tolist(),
        "sample_rate": sample_rate,
        "frequency": 1000,
        "amplitude": 1.0
    }


@pytest.fixture
def multi_channel_waveform_data():
    """Provide multi-channel waveform data for testing."""
    import numpy as np
    
    sample_rate = 8000
    duration = 0.05
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Channel 1: 500 Hz sine
    ch1 = 2.0 * np.sin(2 * np.pi * 500 * t)
    
    # Channel 2: 1500 Hz sine with phase shift
    ch2 = 1.5 * np.sin(2 * np.pi * 1500 * t + np.pi/4)
    
    # Channel 3: Square wave
    ch3 = 1.0 * np.sign(np.sin(2 * np.pi * 200 * t))
    
    # Channel 4: Sawtooth wave
    ch4 = 0.8 * (2 * (t * 300 % 1) - 1)
    
    return {
        "time": t.tolist(),
        "channels": {
            1: ch1.tolist(),
            2: ch2.tolist(),
            3: ch3.tolist(),
            4: ch4.tolist()
        },
        "sample_rate": sample_rate
    }


@pytest.fixture
def protocol_test_signals():
    """Provide test signals for protocol decoding."""
    import numpy as np
    
    sample_rate = 100000  # 100 kHz
    
    # UART test signal (9600 baud, character 'A' = 0x41)
    bit_duration = sample_rate // 9600
    uart_bits = [1] * (bit_duration * 5)  # Idle
    uart_bits += [0] * bit_duration  # Start bit
    # Data bits for 'A' (0x41 = 01000001, LSB first): 1,0,0,0,0,0,1,0
    data_bits = [1, 0, 0, 0, 0, 0, 1, 0]
    for bit in data_bits:
        uart_bits += [bit] * bit_duration
    uart_bits += [1] * bit_duration  # Stop bit
    uart_bits += [1] * (bit_duration * 5)  # Idle
    
    # SPI test signal (8-bit word 0xA5 = 10100101)
    spi_clock = []
    spi_data = []
    for i in range(8):
        bit_val = (0xA5 >> (7 - i)) & 1  # MSB first
        # Clock low
        spi_clock += [0] * 50
        spi_data += [bit_val] * 50
        # Clock high
        spi_clock += [1] * 50
        spi_data += [bit_val] * 50
    
    return {
        "uart": {
            "signal": uart_bits,
            "sample_rate": sample_rate,
            "character": 0x41
        },
        "spi": {
            "clock": spi_clock,
            "data": spi_data,
            "sample_rate": sample_rate,
            "word": 0xA5
        }
    }


@pytest.fixture
def measurement_test_data():
    """Provide test data for measurement analysis."""
    import numpy as np
    
    # Various test signals for different measurements
    sample_rate = 10000
    
    # DC signal for RMS test
    dc_signal = np.ones(1000) * 2.5
    
    # AC sine wave for frequency measurement
    t_ac = np.linspace(0, 0.1, 1000, endpoint=False)
    ac_signal = 3.0 * np.sin(2 * np.pi * 750 * t_ac)
    
    # Step response for rise time measurement
    step_signal = np.concatenate([
        np.zeros(200),  # Initial low
        np.linspace(0, 1, 100),  # Rising edge
        np.ones(700)  # Final high
    ])
    
    # Pulse signal for pulse width measurement
    pulse_signal = np.concatenate([
        np.zeros(300),  # Low
        np.ones(400),   # High pulse
        np.zeros(300)   # Low
    ])
    
    return {
        "dc_signal": dc_signal.tolist(),
        "ac_signal": ac_signal.tolist(),
        "step_signal": step_signal.tolist(),
        "pulse_signal": pulse_signal.tolist(),
        "sample_rate": sample_rate,
        "expected_values": {
            "dc_rms": 2.5,
            "ac_frequency": 750,
            "step_rise_time": 0.01,  # 100 samples at 10kHz
            "pulse_width": 0.04      # 400 samples at 10kHz
        }
    }


@pytest.fixture
def fft_test_data():
    """Provide test data for FFT analysis."""
    import numpy as np
    
    sample_rate = 2048
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Multi-tone signal: 100Hz + 250Hz + 600Hz
    signal = (1.0 * np.sin(2 * np.pi * 100 * t) +
              0.7 * np.sin(2 * np.pi * 250 * t) +
              0.4 * np.sin(2 * np.pi * 600 * t))
    
    # Add small amount of noise
    noise = 0.05 * np.random.randn(len(signal))
    noisy_signal = signal + noise
    
    return {
        "time": t.tolist(),
        "clean_signal": signal.tolist(),
        "noisy_signal": noisy_signal.tolist(),
        "sample_rate": sample_rate,
        "frequencies": [100, 250, 600],
        "amplitudes": [1.0, 0.7, 0.4]
    }


@pytest.fixture
def filter_test_data():
    """Provide test data for filter testing."""
    import numpy as np
    
    sample_rate = 1000
    t = np.linspace(0, 2, 2000, endpoint=False)
    
    # Composite signal: 50Hz + 200Hz + 400Hz
    low_freq = np.sin(2 * np.pi * 50 * t)      # Should pass lowpass
    mid_freq = np.sin(2 * np.pi * 200 * t)     # Target frequency
    high_freq = np.sin(2 * np.pi * 400 * t)    # Should be filtered
    
    composite_signal = low_freq + mid_freq + high_freq
    
    return {
        "time": t.tolist(),
        "composite_signal": composite_signal.tolist(),
        "low_freq_component": low_freq.tolist(),
        "mid_freq_component": mid_freq.tolist(),
        "high_freq_component": high_freq.tolist(),
        "sample_rate": sample_rate,
        "frequencies": [50, 200, 400]
    }


# Pytest markers for test categorization
pytest_markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "hardware: Hardware-dependent tests",
    "slow: Slow tests that take more than 1 second",
    "network: Tests that require network access",
    "simulation: Tests using simulated hardware only"
]


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add appropriate markers."""
    for item in items:
        # Add simulation marker to tests using simulated hardware
        if "simulated_hardware" in getattr(item, "fixturenames", []):
            item.add_marker(pytest.mark.simulation)
        
        # Add integration marker to tests using full server
        if "mcp_server" in getattr(item, "fixturenames", []):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.name.lower() for keyword in ["comprehensive", "multi_channel", "real_time"]):
            item.add_marker(pytest.mark.slow)


# Skip marks for missing dependencies
def pytest_runtest_setup(item):
    """Setup function to skip tests based on dependencies."""
    # Skip hardware tests if hardware dependencies are missing
    if item.get_closest_marker("hardware"):
        try:
            import serial
            import sounddevice
        except ImportError:
            pytest.skip("Hardware dependencies not available")
    
    # Skip network tests if network is not available
    if item.get_closest_marker("network"):
        import socket
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
        except OSError:
            pytest.skip("Network not available")