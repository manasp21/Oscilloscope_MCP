"""
Test ADC interface functionality.

Tests for universal ADC interface, data parsers, and channel configurations.
"""

import pytest
import json
import struct
from typing import Dict, Any

from oscilloscope_mcp.hardware.adc_interface import (
    UniversalADCInterface,
    ADCChannelConfig,
    ADCDataFormat,
    RawBinaryParser,
    JSONArrayParser,
    ASCIICSVParser,
    AudioFloat32Parser
)


class TestADCChannelConfig:
    """Test ADC channel configuration."""

    def test_channel_creation(self):
        """Test basic channel configuration creation."""
        config = ADCChannelConfig(
            channel_id=0,
            voltage_range=5.0,
            resolution_bits=12
        )
        
        assert config.channel_id == 0
        assert config.voltage_range == 5.0
        assert config.resolution_bits == 12
        assert config.coupling == "DC"
        assert config.impedance == "1M"
        assert config.enabled is True

    def test_voltage_conversion(self):
        """Test raw value to voltage conversion."""
        config = ADCChannelConfig(
            channel_id=0,
            voltage_range=3.3,
            resolution_bits=12
        )
        
        # Test zero voltage
        assert config.raw_to_voltage(0) == pytest.approx(-1.65, abs=0.01)
        
        # Test full scale
        assert config.raw_to_voltage(4095) == pytest.approx(1.65, abs=0.01)
        
        # Test mid scale
        assert config.raw_to_voltage(2048) == pytest.approx(0.0, abs=0.01)

    def test_config_serialization(self):
        """Test configuration serialization to dictionary."""
        config = ADCChannelConfig(
            channel_id=1,
            voltage_range=10.0,
            resolution_bits=16,
            coupling="AC",
            impedance="50"
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["channel_id"] == 1
        assert config_dict["voltage_range"] == 10.0
        assert config_dict["resolution_bits"] == 16
        assert config_dict["coupling"] == "AC"
        assert config_dict["impedance"] == "50"


class TestUniversalADCInterface:
    """Test universal ADC interface."""

    def test_interface_creation(self):
        """Test ADC interface creation."""
        adc = UniversalADCInterface("test_device")
        
        assert adc.device_id == "test_device"
        assert adc.is_active is False
        assert adc.sample_rate == 1000.0
        assert len(adc.channels) == 0

    def test_channel_management(self):
        """Test adding and managing channels."""
        adc = UniversalADCInterface("test_device")
        
        # Add first channel
        config1 = adc.add_channel(0, voltage_range=5.0, resolution_bits=12)
        assert config1.channel_id == 0
        assert len(adc.channels) == 1
        
        # Add second channel
        config2 = adc.add_channel(1, voltage_range=3.3, resolution_bits=10)
        assert config2.channel_id == 1
        assert len(adc.channels) == 2
        
        # Verify channels are stored correctly
        assert adc.channels[0] == config1
        assert adc.channels[1] == config2

    def test_data_format_configuration(self):
        """Test data format configuration."""
        adc = UniversalADCInterface("test_device")
        adc.add_channel(0, voltage_range=5.0, resolution_bits=12)
        
        # Test JSON format
        adc.set_data_format(ADCDataFormat.JSON_ARRAY)
        assert adc.data_format == ADCDataFormat.JSON_ARRAY
        assert adc.parser is not None
        
        # Test raw binary format
        adc.set_data_format(
            ADCDataFormat.RAW_BINARY,
            parser_params={
                "bytes_per_sample": 2,
                "num_channels": 1,
                "byte_order": "little"
            }
        )
        assert adc.data_format == ADCDataFormat.RAW_BINARY
        assert isinstance(adc.parser, RawBinaryParser)

    async def test_data_processing(self):
        """Test raw data processing."""
        adc = UniversalADCInterface("test_device")
        adc.add_channel(0, voltage_range=5.0, resolution_bits=12)
        adc.set_data_format(ADCDataFormat.JSON_ARRAY)
        
        # Create test JSON data
        test_data = {
            "channels": {
                "0": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }
        json_bytes = json.dumps(test_data).encode('utf-8')
        
        # Process the data
        result = await adc.process_raw_data(json_bytes)
        
        assert result is not None
        assert 0 in result
        assert len(result[0]) == 5
        assert result[0] == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_statistics(self):
        """Test interface statistics."""
        adc = UniversalADCInterface("test_device")
        adc.add_channel(0, voltage_range=5.0, resolution_bits=12)
        
        stats = adc.get_statistics()
        
        assert stats["device_id"] == "test_device"
        assert stats["is_active"] is False
        assert stats["total_samples"] == 0
        assert stats["total_bytes"] == 0
        assert stats["sample_rate"] == 1000.0
        assert 0 in stats["channels"]


class TestDataParsers:
    """Test ADC data parsers."""

    @pytest.fixture
    def test_channels(self):
        """Create test channel configurations."""
        return {
            0: ADCChannelConfig(0, voltage_range=5.0, resolution_bits=12),
            1: ADCChannelConfig(1, voltage_range=3.3, resolution_bits=10),
        }

    async def test_json_array_parser(self, test_channels):
        """Test JSON array data parser."""
        parser = JSONArrayParser()
        
        test_data = {
            "channels": {
                "0": [0.1, 0.2, 0.3],
                "1": [1.0, 1.1, 1.2]
            }
        }
        json_bytes = json.dumps(test_data).encode('utf-8')
        
        result = await parser.parse(json_bytes, test_channels)
        
        assert 0 in result
        assert 1 in result
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [1.0, 1.1, 1.2]

    async def test_raw_binary_parser(self, test_channels):
        """Test raw binary data parser."""
        parser = RawBinaryParser(
            bytes_per_sample=2,
            num_channels=2,
            byte_order="little"
        )
        
        # Create test binary data (2 channels, 3 samples each, 2 bytes per sample)
        # Sample values: CH0=[1000, 2000, 3000], CH1=[500, 1500, 2500]
        raw_data = struct.pack('<HHHHHH', 1000, 500, 2000, 1500, 3000, 2500)
        
        result = await parser.parse(raw_data, test_channels)
        
        assert 0 in result
        assert 1 in result
        assert len(result[0]) == 3
        assert len(result[1]) == 3
        
        # Verify voltage conversion (approximate due to floating point)
        assert result[0][0] == pytest.approx(test_channels[0].raw_to_voltage(1000), abs=0.01)
        assert result[1][0] == pytest.approx(test_channels[1].raw_to_voltage(500), abs=0.01)

    async def test_ascii_csv_parser(self, test_channels):
        """Test ASCII CSV data parser."""
        parser = ASCIICSVParser()
        
        # CSV format: channel_id,value
        csv_data = "0,0.5\n1,1.0\n0,1.5\n1,2.0\n0,2.5\n1,3.0\n"
        csv_bytes = csv_data.encode('utf-8')
        
        result = await parser.parse(csv_bytes, test_channels)
        
        assert 0 in result
        assert 1 in result
        assert result[0] == [0.5, 1.5, 2.5]
        assert result[1] == [1.0, 2.0, 3.0]

    async def test_audio_float32_parser(self, test_channels):
        """Test audio float32 data parser."""
        parser = AudioFloat32Parser()
        
        # Create test audio data (stereo, 3 samples)
        samples = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]  # L,R,L,R,L,R
        audio_data = struct.pack('f' * len(samples), *samples)
        
        result = await parser.parse(audio_data, test_channels)
        
        assert 0 in result  # Left channel
        assert 1 in result  # Right channel
        assert len(result[0]) == 3
        assert len(result[1]) == 3
        
        # Verify stereo separation
        assert result[0] == pytest.approx([0.1, 0.3, 0.5], abs=0.001)
        assert result[1] == pytest.approx([-0.2, -0.4, -0.6], abs=0.001)


class TestADCIntegration:
    """Integration tests for ADC interface."""

    async def test_end_to_end_json_processing(self):
        """Test complete JSON data processing workflow."""
        adc = UniversalADCInterface("integration_test")
        
        # Setup channels
        adc.add_channel(0, voltage_range=5.0, resolution_bits=12)
        adc.add_channel(1, voltage_range=3.3, resolution_bits=10)
        
        # Configure for JSON
        adc.set_data_format(ADCDataFormat.JSON_ARRAY)
        
        # Process test data
        test_data = {
            "channels": {
                "0": [0.0, 1.0, 2.0, 3.0, 4.0],
                "1": [0.5, 1.5, 2.5, 3.0, 3.2]
            }
        }
        json_bytes = json.dumps(test_data).encode('utf-8')
        
        result = await adc.process_raw_data(json_bytes)
        
        assert result is not None
        assert len(result) == 2
        assert result[0] == test_data["channels"]["0"]
        assert result[1] == test_data["channels"]["1"]
        
        # Verify statistics updated
        stats = adc.get_statistics()
        assert stats["total_samples"] == 5
        assert stats["total_bytes"] == len(json_bytes)

    async def test_end_to_end_binary_processing(self):
        """Test complete binary data processing workflow."""
        adc = UniversalADCInterface("binary_test")
        
        # Setup single channel
        adc.add_channel(0, voltage_range=5.0, resolution_bits=12)
        
        # Configure for raw binary
        adc.set_data_format(
            ADCDataFormat.RAW_BINARY,
            parser_params={
                "bytes_per_sample": 2,
                "num_channels": 1,
                "byte_order": "little"
            }
        )
        
        # Create binary test data
        raw_values = [0, 1024, 2048, 3072, 4095]  # Various ADC values
        binary_data = struct.pack('<' + 'H' * len(raw_values), *raw_values)
        
        result = await adc.process_raw_data(binary_data)
        
        assert result is not None
        assert 0 in result
        assert len(result[0]) == 5
        
        # Verify voltage conversion
        channel_config = adc.channels[0]
        expected_voltages = [channel_config.raw_to_voltage(val) for val in raw_values]
        assert result[0] == pytest.approx(expected_voltages, abs=0.001)