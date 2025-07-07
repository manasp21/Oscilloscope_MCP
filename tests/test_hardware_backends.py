"""
Test hardware backend implementations.

Tests for all hardware backends including simulation, USB, Ethernet, and PCIe stubs.
"""

import pytest
from typing import Dict, Any

from oscilloscope_mcp.hardware.simulation import SimulatedHardware
from oscilloscope_mcp.hardware.usb_backend import USBHardware
from oscilloscope_mcp.hardware.ethernet_backend import EthernetHardware
from oscilloscope_mcp.hardware.pcie_backend import PCIeHardware


@pytest.fixture
def basic_config() -> Dict[str, Any]:
    """Basic configuration for hardware testing."""
    return {
        "hardware_interface": "simulation",
        "sample_rate": 1e6,
        "channels": 4,
        "buffer_size": 1000,
        "timeout": 5.0,
    }


class TestSimulatedHardware:
    """Test simulated hardware backend."""

    async def test_initialization(self, basic_config):
        """Test hardware initialization."""
        hardware = SimulatedHardware()
        await hardware.initialize(basic_config)
        assert hardware.is_initialized is True
        await hardware.cleanup()

    async def test_oscilloscope_operations(self, basic_config):
        """Test oscilloscope functionality."""
        hardware = SimulatedHardware()
        await hardware.initialize(basic_config)
        
        oscilloscope = await hardware.get_oscilloscope()
        
        # Test channel configuration
        result = await oscilloscope.configure_channels([1, 2], 5.0, "DC", "1M")
        assert result["status"] == "success"
        
        # Test timebase configuration
        result = await oscilloscope.set_timebase(1e6, 1000)
        assert result["status"] == "success"
        
        # Test trigger setup
        result = await oscilloscope.setup_trigger("CH1", "edge", 0.5, "rising", 0.0)
        assert result["status"] == "success"
        
        # Test waveform acquisition
        waveform_data = await oscilloscope.acquire_waveform([1, 2], 5.0)
        assert "channels" in waveform_data
        assert len(waveform_data["channels"]) == 2
        
        await hardware.cleanup()

    async def test_function_generator_operations(self, basic_config):
        """Test function generator functionality."""
        hardware = SimulatedHardware()
        await hardware.initialize(basic_config)
        
        function_gen = await hardware.get_function_generator()
        
        # Test standard waveform generation
        result = await function_gen.generate_standard_waveform(
            1, "sine", 1000, 1.0, 0.0, 0.0
        )
        assert result["status"] == "success"
        
        # Test arbitrary waveform generation
        samples = [0.1, 0.2, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.2, -0.1]
        result = await function_gen.generate_arbitrary_waveform(
            1, samples, 1000, 1.0, 0.0
        )
        assert result["status"] == "success"
        
        await hardware.cleanup()


class TestUSBHardware:
    """Test USB hardware backend stubs."""

    async def test_usb_hardware_fallback(self, basic_config):
        """Test that USB hardware falls back to simulation."""
        hardware = USBHardware()
        await hardware.initialize(basic_config)
        
        # Should initialize successfully using simulation
        assert hardware.is_initialized is True
        
        # Should provide oscilloscope functionality
        oscilloscope = await hardware.get_oscilloscope()
        result = await oscilloscope.configure_channels([1], 5.0, "DC", "1M")
        assert result["status"] == "success"
        
        await hardware.cleanup()

    async def test_usb_components_exist(self):
        """Test that USB components can be instantiated."""
        hardware = USBHardware()
        assert hardware.oscilloscope is not None
        assert hardware.function_generator is not None


class TestEthernetHardware:
    """Test Ethernet hardware backend stubs."""

    async def test_ethernet_hardware_fallback(self, basic_config):
        """Test that Ethernet hardware falls back to simulation."""
        hardware = EthernetHardware()
        await hardware.initialize(basic_config)
        
        assert hardware.is_initialized is True
        
        # Test function generator functionality through ethernet backend
        function_gen = await hardware.get_function_generator()
        result = await function_gen.generate_standard_waveform(
            1, "square", 500, 2.0, 0.5, 0.0
        )
        assert result["status"] == "success"
        
        await hardware.cleanup()

    async def test_ethernet_components_exist(self):
        """Test that Ethernet components can be instantiated."""
        hardware = EthernetHardware()
        assert hardware.oscilloscope is not None
        assert hardware.function_generator is not None


class TestPCIeHardware:
    """Test PCIe hardware backend stubs."""

    async def test_pcie_hardware_fallback(self, basic_config):
        """Test that PCIe hardware falls back to simulation."""
        hardware = PCIeHardware()
        await hardware.initialize(basic_config)
        
        assert hardware.is_initialized is True
        
        # Test oscilloscope functionality through pcie backend
        oscilloscope = await hardware.get_oscilloscope()
        waveform_data = await oscilloscope.acquire_waveform([1, 2, 3], 10.0)
        assert "channels" in waveform_data
        assert len(waveform_data["channels"]) == 3
        
        await hardware.cleanup()

    async def test_pcie_components_exist(self):
        """Test that PCIe components can be instantiated."""
        hardware = PCIeHardware()
        assert hardware.oscilloscope is not None
        assert hardware.function_generator is not None


class TestHardwareIntegration:
    """Integration tests across hardware backends."""

    @pytest.mark.parametrize("hardware_class", [
        SimulatedHardware,
        USBHardware, 
        EthernetHardware,
        PCIeHardware
    ])
    async def test_all_backends_compatibility(self, hardware_class, basic_config):
        """Test that all hardware backends provide compatible interfaces."""
        hardware = hardware_class()
        await hardware.initialize(basic_config)
        
        # All backends should provide oscilloscope
        oscilloscope = await hardware.get_oscilloscope()
        assert oscilloscope is not None
        
        # All backends should provide function generator
        function_gen = await hardware.get_function_generator()
        assert function_gen is not None
        
        # All backends should support cleanup
        await hardware.cleanup()
        assert hardware.is_initialized is False

    async def test_hardware_status_consistency(self, basic_config):
        """Test that hardware status is consistent across backends."""
        backends = [
            SimulatedHardware(),
            USBHardware(),
            EthernetHardware(),
            PCIeHardware()
        ]
        
        for hardware in backends:
            await hardware.initialize(basic_config)
            
            oscilloscope = await hardware.get_oscilloscope()
            status = await oscilloscope.get_status()
            
            # All backends should return status in consistent format
            assert "status" in status
            assert "channels" in status
            assert "sample_rate" in status
            
            await hardware.cleanup()