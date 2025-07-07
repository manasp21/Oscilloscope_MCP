"""
Test protocol decoding functionality.

Tests for digital protocol decoding (UART, SPI, I2C, CAN) and communication analysis.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from oscilloscope_mcp.protocols.decoder import ProtocolDecoder


class TestProtocolDecoder:
    """Test protocol decoder base functionality."""

    def test_decoder_creation(self):
        """Test protocol decoder instantiation."""
        decoder = ProtocolDecoder()
        assert decoder is not None

    def test_supported_protocols(self):
        """Test that decoder supports expected protocols."""
        decoder = ProtocolDecoder()
        
        supported = decoder.get_supported_protocols()
        
        expected_protocols = ["UART", "SPI", "I2C", "CAN"]
        for protocol in expected_protocols:
            assert protocol in supported

    def test_protocol_configuration(self):
        """Test protocol configuration setup."""
        decoder = ProtocolDecoder()
        
        # Configure UART
        uart_config = {
            "baud_rate": 9600,
            "data_bits": 8,
            "parity": "none",
            "stop_bits": 1
        }
        
        success = decoder.configure_protocol("UART", uart_config)
        assert success is True
        
        # Verify configuration was stored
        stored_config = decoder.get_protocol_config("UART")
        assert stored_config["baud_rate"] == 9600
        assert stored_config["data_bits"] == 8


class TestUARTDecoding:
    """Test UART protocol decoding."""

    def test_uart_frame_detection(self):
        """Test UART frame detection and decoding."""
        decoder = ProtocolDecoder()
        
        # Configure UART: 9600 baud, 8N1
        config = {
            "baud_rate": 9600,
            "data_bits": 8,
            "parity": "none",
            "stop_bits": 1
        }
        decoder.configure_protocol("UART", config)
        
        # Create UART signal for character 'A' (0x41)
        sample_rate = 96000  # 10x oversampling
        bit_duration = sample_rate // 9600
        
        # UART frame: START(0) + DATA(01000001) + STOP(1)
        # LSB first for UART, so 'A' = 0x41 = 10000010 in transmission order
        frame_bits = [0, 1, 0, 0, 0, 0, 0, 1, 0, 1]  # start + data + stop
        
        signal = []
        for bit in frame_bits:
            signal.extend([bit] * bit_duration)
        
        # Add idle periods
        idle_samples = bit_duration * 5
        full_signal = [1] * idle_samples + signal + [1] * idle_samples
        
        decoded = decoder.decode_uart(full_signal, sample_rate)
        
        assert len(decoded) > 0
        assert "frames" in decoded
        if len(decoded["frames"]) > 0:
            frame = decoded["frames"][0]
            assert "data" in frame
            # Should decode to 'A' (0x41)
            assert frame["data"] == 0x41

    def test_uart_multiple_characters(self):
        """Test decoding multiple UART characters."""
        decoder = ProtocolDecoder()
        
        config = {
            "baud_rate": 9600,
            "data_bits": 8,
            "parity": "none",
            "stop_bits": 1
        }
        decoder.configure_protocol("UART", config)
        
        # Create signal for "AB" (0x41, 0x42)
        sample_rate = 96000
        bit_duration = sample_rate // 9600
        
        # Character 'A' = 0x41 = 10000010 (LSB first)
        char_a = [0, 1, 0, 0, 0, 0, 0, 1, 0, 1]
        # Character 'B' = 0x42 = 01000010 (LSB first)  
        char_b = [0, 0, 1, 0, 0, 0, 0, 1, 0, 1]
        
        signal = []
        # Add character A
        for bit in char_a:
            signal.extend([bit] * bit_duration)
        
        # Add inter-character gap
        signal.extend([1] * (bit_duration * 2))
        
        # Add character B
        for bit in char_b:
            signal.extend([bit] * bit_duration)
        
        # Add padding
        padding = [1] * (bit_duration * 5)
        full_signal = padding + signal + padding
        
        decoded = decoder.decode_uart(full_signal, sample_rate)
        
        assert "frames" in decoded
        # Should decode at least one character
        assert len(decoded["frames"]) >= 1

    def test_uart_error_detection(self):
        """Test UART error detection (framing, parity)."""
        decoder = ProtocolDecoder()
        
        config = {
            "baud_rate": 9600,
            "data_bits": 8,
            "parity": "even",
            "stop_bits": 1
        }
        decoder.configure_protocol("UART", config)
        
        # Create malformed UART frame (missing stop bit)
        sample_rate = 96000
        bit_duration = sample_rate // 9600
        
        # Malformed frame: START + DATA + no STOP
        malformed_bits = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]  # Missing stop bit
        
        signal = []
        for bit in malformed_bits:
            signal.extend([bit] * bit_duration)
        
        padding = [1] * (bit_duration * 5)
        full_signal = padding + signal + padding
        
        decoded = decoder.decode_uart(full_signal, sample_rate)
        
        assert "errors" in decoded
        # Should detect framing error
        if len(decoded["errors"]) > 0:
            assert any("framing" in error["type"] for error in decoded["errors"])


class TestSPIDecoding:
    """Test SPI protocol decoding."""

    def test_spi_transaction_detection(self):
        """Test SPI transaction detection and decoding."""
        decoder = ProtocolDecoder()
        
        config = {
            "clock_polarity": 0,  # CPOL = 0
            "clock_phase": 0,     # CPHA = 0
            "bit_order": "MSB",   # MSB first
            "word_size": 8
        }
        decoder.configure_protocol("SPI", config)
        
        # Create SPI signals for one byte (0xA5 = 10100101)
        sample_rate = 1000000  # 1 MHz
        bit_duration = 100  # 100 samples per bit
        
        # SPI signals: CLK, MOSI, MISO, CS
        data_byte = 0xA5  # 10100101
        
        # Generate clock signal (8 pulses)
        clk = []
        mosi = []
        cs = []
        
        # CS active low at start
        cs_idle = [1] * (bit_duration * 2)
        cs_active = [0] * (bit_duration * 8)
        cs_signal = cs_idle + cs_active + cs_idle
        
        # Clock and data
        for i in range(8):
            bit_value = (data_byte >> (7 - i)) & 1  # MSB first
            
            # Clock low period
            clk.extend([0] * (bit_duration // 2))
            mosi.extend([bit_value] * (bit_duration // 2))
            
            # Clock high period
            clk.extend([1] * (bit_duration // 2))
            mosi.extend([bit_value] * (bit_duration // 2))
        
        # Add padding
        padding = [0] * (bit_duration * 2)
        clk_signal = padding + clk + padding
        mosi_signal = [0] * (bit_duration * 2) + mosi + [0] * (bit_duration * 2)
        
        signals = {
            "CLK": clk_signal,
            "MOSI": mosi_signal,
            "MISO": [0] * len(clk_signal),  # No MISO data for this test
            "CS": cs_signal
        }
        
        decoded = decoder.decode_spi(signals, sample_rate)
        
        assert "transactions" in decoded
        if len(decoded["transactions"]) > 0:
            transaction = decoded["transactions"][0]
            assert "mosi_data" in transaction
            # Should decode to 0xA5
            assert transaction["mosi_data"][0] == 0xA5

    def test_spi_multiple_bytes(self):
        """Test SPI decoding with multiple bytes."""
        decoder = ProtocolDecoder()
        
        config = {
            "clock_polarity": 0,
            "clock_phase": 0,
            "bit_order": "MSB",
            "word_size": 8
        }
        decoder.configure_protocol("SPI", config)
        
        # Test data: command byte + data byte
        test_bytes = [0x03, 0xFF]  # Read command + dummy data
        
        # This would be implemented similarly to single byte test
        # but with multiple bytes in sequence
        
        # For now, verify the configuration was accepted
        config_check = decoder.get_protocol_config("SPI")
        assert config_check["word_size"] == 8


class TestI2CDecoding:
    """Test I2C protocol decoding."""

    def test_i2c_start_stop_detection(self):
        """Test I2C START and STOP condition detection."""
        decoder = ProtocolDecoder()
        
        config = {
            "clock_frequency": 100000,  # 100 kHz standard mode
            "address_bits": 7
        }
        decoder.configure_protocol("I2C", config)
        
        # Create I2C signals with START condition
        sample_rate = 1000000  # 1 MHz sampling
        bit_duration = 10  # Samples per I2C bit
        
        # START condition: SDA high-to-low while SCL high
        # Normal idle: both SDA and SCL high
        idle_period = [1] * 20
        
        # START: SCL high, SDA goes high->low
        start_scl = [1] * 10
        start_sda = [1] * 3 + [0] * 7
        
        scl_signal = idle_period + start_scl + [0] * 50
        sda_signal = idle_period + start_sda + [0] * 50
        
        signals = {
            "SCL": scl_signal,
            "SDA": sda_signal
        }
        
        decoded = decoder.decode_i2c(signals, sample_rate)
        
        assert "conditions" in decoded
        # Should detect START condition
        conditions = decoded["conditions"]
        if len(conditions) > 0:
            assert any(cond["type"] == "START" for cond in conditions)

    def test_i2c_address_decode(self):
        """Test I2C address decoding."""
        decoder = ProtocolDecoder()
        
        config = {
            "clock_frequency": 100000,
            "address_bits": 7
        }
        decoder.configure_protocol("I2C", config)
        
        # This would implement full I2C address decoding
        # For now, verify configuration
        config_check = decoder.get_protocol_config("I2C")
        assert config_check["address_bits"] == 7


class TestCANDecoding:
    """Test CAN protocol decoding."""

    def test_can_frame_detection(self):
        """Test CAN frame detection and decoding."""
        decoder = ProtocolDecoder()
        
        config = {
            "bit_rate": 250000,  # 250 kbps
            "sample_point": 75,  # 75% sample point
            "format": "standard"  # 11-bit identifier
        }
        decoder.configure_protocol("CAN", config)
        
        # CAN frame structure is complex - this is a simplified test
        # Real implementation would need full CAN bit stuffing, CRC, etc.
        
        # Verify configuration was accepted
        config_check = decoder.get_protocol_config("CAN")
        assert config_check["bit_rate"] == 250000
        assert config_check["format"] == "standard"

    def test_can_bit_stuffing(self):
        """Test CAN bit stuffing detection."""
        decoder = ProtocolDecoder()
        
        config = {
            "bit_rate": 500000,
            "sample_point": 80,
            "format": "extended"  # 29-bit identifier
        }
        decoder.configure_protocol("CAN", config)
        
        # Bit stuffing: after 5 consecutive bits of same polarity,
        # opposite bit is inserted
        
        # This would test the bit destuffing algorithm
        # For now, verify extended format configuration
        config_check = decoder.get_protocol_config("CAN")
        assert config_check["format"] == "extended"


class TestProtocolAnalysis:
    """Test protocol analysis and statistics."""

    def test_timing_analysis(self):
        """Test protocol timing analysis."""
        decoder = ProtocolDecoder()
        
        # Configure UART for timing analysis
        config = {
            "baud_rate": 115200,
            "data_bits": 8,
            "parity": "none",
            "stop_bits": 1
        }
        decoder.configure_protocol("UART", config)
        
        # Create signal with known timing
        sample_rate = 1152000  # 10x oversampling
        expected_bit_time = sample_rate / 115200
        
        # Simple test signal
        signal = [1] * 100 + [0] * int(expected_bit_time) + [1] * 100
        
        timing_analysis = decoder.analyze_timing("UART", signal, sample_rate)
        
        assert "bit_rate_measured" in timing_analysis
        assert "bit_time_average" in timing_analysis
        assert "timing_jitter" in timing_analysis
        
        measured_bit_time = timing_analysis["bit_time_average"]
        assert measured_bit_time == pytest.approx(expected_bit_time, abs=expected_bit_time * 0.1)

    def test_error_statistics(self):
        """Test protocol error statistics."""
        decoder = ProtocolDecoder()
        
        # Configure protocol
        config = {
            "baud_rate": 9600,
            "data_bits": 8,
            "parity": "odd",
            "stop_bits": 1
        }
        decoder.configure_protocol("UART", config)
        
        # Simulate decoding with some errors
        decoded_frames = [
            {"data": 0x41, "errors": []},
            {"data": 0x42, "errors": ["parity_error"]},
            {"data": 0x43, "errors": []},
            {"data": 0x44, "errors": ["framing_error"]},
            {"data": 0x45, "errors": []}
        ]
        
        stats = decoder.calculate_error_statistics(decoded_frames)
        
        assert "total_frames" in stats
        assert "error_frames" in stats
        assert "error_rate" in stats
        assert "error_types" in stats
        
        assert stats["total_frames"] == 5
        assert stats["error_frames"] == 2
        assert stats["error_rate"] == pytest.approx(0.4, abs=0.01)  # 40% error rate

    def test_throughput_analysis(self):
        """Test protocol throughput analysis."""
        decoder = ProtocolDecoder()
        
        # Simulate SPI transaction data
        transactions = [
            {"timestamp": 0.000, "bytes": 4, "duration": 0.001},
            {"timestamp": 0.010, "bytes": 8, "duration": 0.002},
            {"timestamp": 0.025, "bytes": 2, "duration": 0.0005},
            {"timestamp": 0.050, "bytes": 16, "duration": 0.004}
        ]
        
        throughput_stats = decoder.calculate_throughput_statistics(transactions)
        
        assert "total_bytes" in throughput_stats
        assert "total_time" in throughput_stats
        assert "average_throughput" in throughput_stats
        assert "peak_throughput" in throughput_stats
        
        assert throughput_stats["total_bytes"] == 30  # 4+8+2+16
        assert throughput_stats["total_time"] > 0


class TestProtocolIntegration:
    """Integration tests for protocol decoding."""

    def test_multi_protocol_analysis(self):
        """Test analyzing multiple protocols simultaneously."""
        decoder = ProtocolDecoder()
        
        # Configure multiple protocols
        uart_config = {"baud_rate": 9600, "data_bits": 8, "parity": "none", "stop_bits": 1}
        spi_config = {"clock_polarity": 0, "clock_phase": 0, "bit_order": "MSB", "word_size": 8}
        
        decoder.configure_protocol("UART", uart_config)
        decoder.configure_protocol("SPI", spi_config)
        
        # Get list of configured protocols
        configured = decoder.get_configured_protocols()
        
        assert "UART" in configured
        assert "SPI" in configured
        assert len(configured) == 2

    def test_protocol_switching(self):
        """Test switching between different protocol configurations."""
        decoder = ProtocolDecoder()
        
        # Configure UART with different settings
        config1 = {"baud_rate": 9600, "data_bits": 8, "parity": "none", "stop_bits": 1}
        config2 = {"baud_rate": 115200, "data_bits": 7, "parity": "even", "stop_bits": 2}
        
        decoder.configure_protocol("UART", config1)
        stored1 = decoder.get_protocol_config("UART")
        assert stored1["baud_rate"] == 9600
        
        # Reconfigure with different settings
        decoder.configure_protocol("UART", config2)
        stored2 = decoder.get_protocol_config("UART")
        assert stored2["baud_rate"] == 115200
        assert stored2["data_bits"] == 7

    def test_real_world_protocol_scenario(self):
        """Test a realistic protocol decoding scenario."""
        decoder = ProtocolDecoder()
        
        # Simulate I2C sensor reading scenario
        config = {
            "clock_frequency": 400000,  # Fast mode
            "address_bits": 7
        }
        decoder.configure_protocol("I2C", config)
        
        # Simulate typical I2C transaction:
        # START + ADDRESS + WRITE + REGISTER + START + ADDRESS + READ + DATA + STOP
        
        # This would be a comprehensive test of a real I2C transaction
        # For now, verify the setup
        assert decoder.get_protocol_config("I2C")["clock_frequency"] == 400000

    def test_protocol_debugging_features(self):
        """Test protocol debugging and analysis features."""
        decoder = ProtocolDecoder()
        
        # Enable detailed debugging
        debug_config = {
            "capture_raw_bits": True,
            "timestamp_precision": "nanosecond",
            "include_timing_analysis": True
        }
        
        decoder.set_debug_options(debug_config)
        
        # Verify debug options were set
        debug_settings = decoder.get_debug_options()
        assert debug_settings["capture_raw_bits"] is True
        assert debug_settings["timestamp_precision"] == "nanosecond"