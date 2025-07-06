"""
Universal ADC Interface for Hardware Integration.

This module provides a standardized interface for connecting any ADC hardware
to the MCP oscilloscope server. It supports various connection types and
automatically converts different data formats to the standard MCP format.

Supported ADC Sources:
- Audio interfaces (microphones, line inputs)
- USB ADCs (via serial or HID protocols) 
- SPI ADCs (via GPIO or USB-SPI bridges)
- I2C ADCs (via GPIO or USB-I2C bridges)
- Ethernet ADCs (TCP/UDP data streams)
- PCIe ADCs (via kernel drivers)
- Custom protocols (user-defined parsers)
"""

import asyncio
import json
import time
import struct
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import numpy as np
import structlog

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

logger = structlog.get_logger(__name__)


class ADCDataFormat(Enum):
    """Supported ADC data formats."""
    RAW_BINARY = "raw_binary"           # Raw bytes
    ASCII_CSV = "ascii_csv"             # CSV text format
    JSON_ARRAY = "json_array"           # JSON arrays
    STRUCT_PACKED = "struct_packed"     # Python struct format
    AUDIO_FLOAT32 = "audio_float32"     # 32-bit float audio
    AUDIO_INT16 = "audio_int16"         # 16-bit integer audio
    CUSTOM = "custom"                   # Custom parser function


class ADCChannelConfig:
    """Configuration for a single ADC channel."""
    
    def __init__(
        self,
        channel_id: int,
        voltage_range: float = 3.3,
        resolution_bits: int = 12,
        coupling: str = "DC",
        impedance: str = "1M",
        gain: float = 1.0,
        offset: float = 0.0,
        calibration_slope: float = 1.0,
        calibration_offset: float = 0.0,
        enabled: bool = True
    ):
        self.channel_id = channel_id
        self.voltage_range = voltage_range
        self.resolution_bits = resolution_bits
        self.coupling = coupling
        self.impedance = impedance
        self.gain = gain
        self.offset = offset
        self.calibration_slope = calibration_slope
        self.calibration_offset = calibration_offset
        self.enabled = enabled
        
        # Calculated values
        self.max_raw_value = (2 ** resolution_bits) - 1
        self.voltage_per_bit = voltage_range / self.max_raw_value
    
    def raw_to_voltage(self, raw_value: Union[int, float]) -> float:
        """Convert raw ADC value to calibrated voltage."""
        # Apply gain and offset
        adjusted = (raw_value * self.gain) + self.offset
        
        # Convert to voltage
        voltage = adjusted * self.voltage_per_bit
        
        # Apply calibration
        calibrated = (voltage * self.calibration_slope) + self.calibration_offset
        
        return calibrated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "channel_id": self.channel_id,
            "voltage_range": self.voltage_range,
            "resolution_bits": self.resolution_bits,
            "coupling": self.coupling,
            "impedance": self.impedance,
            "gain": self.gain,
            "offset": self.offset,
            "calibration_slope": self.calibration_slope,
            "calibration_offset": self.calibration_offset,
            "enabled": self.enabled,
            "max_raw_value": self.max_raw_value,
            "voltage_per_bit": self.voltage_per_bit
        }


class ADCDataParser(ABC):
    """Abstract base class for ADC data parsers."""
    
    @abstractmethod
    async def parse(self, raw_data: bytes, channels: Dict[int, ADCChannelConfig]) -> Dict[int, List[float]]:
        """Parse raw data and return voltage values by channel."""
        pass


class RawBinaryParser(ADCDataParser):
    """Parser for raw binary ADC data."""
    
    def __init__(self, bytes_per_sample: int, num_channels: int, byte_order: str = 'little'):
        self.bytes_per_sample = bytes_per_sample
        self.num_channels = num_channels
        self.byte_order = byte_order
        
        # Determine struct format
        if bytes_per_sample == 1:
            self.format_char = 'B'  # unsigned byte
        elif bytes_per_sample == 2:
            self.format_char = 'H' if byte_order == 'little' else '>H'  # unsigned short
        elif bytes_per_sample == 4:
            self.format_char = 'I' if byte_order == 'little' else '>I'  # unsigned int
        else:
            raise ValueError(f"Unsupported bytes_per_sample: {bytes_per_sample}")
    
    async def parse(self, raw_data: bytes, channels: Dict[int, ADCChannelConfig]) -> Dict[int, List[float]]:
        """Parse raw binary data."""
        total_bytes_per_sample = self.bytes_per_sample * self.num_channels
        num_samples = len(raw_data) // total_bytes_per_sample
        
        if num_samples == 0:
            return {}
        
        # Unpack binary data
        format_string = f"{self.format_char}" * (self.num_channels * num_samples)
        raw_values = struct.unpack(format_string, raw_data[:num_samples * total_bytes_per_sample])
        
        # Organize by channel
        channel_data = {}
        for ch_id, config in channels.items():
            if not config.enabled or ch_id >= self.num_channels:
                continue
                
            # Extract samples for this channel
            channel_samples = []
            for sample_idx in range(num_samples):
                raw_idx = sample_idx * self.num_channels + ch_id
                if raw_idx < len(raw_values):
                    raw_value = raw_values[raw_idx]
                    voltage = config.raw_to_voltage(raw_value)
                    channel_samples.append(voltage)
            
            channel_data[ch_id] = channel_samples
        
        return channel_data


class AudioFloat32Parser(ADCDataParser):
    """Parser for 32-bit float audio data (like from sound devices)."""
    
    async def parse(self, raw_data: bytes, channels: Dict[int, ADCChannelConfig]) -> Dict[int, List[float]]:
        """Parse float32 audio data."""
        # Convert bytes to float32 array
        float_array = np.frombuffer(raw_data, dtype=np.float32)
        
        # Determine number of channels from data shape
        if len(channels) == 1:
            # Mono audio
            channel_data = {}
            for ch_id, config in channels.items():
                if config.enabled:
                    # Scale audio (-1 to 1) to voltage range
                    voltage_data = float_array * (config.voltage_range / 2.0)
                    channel_data[ch_id] = voltage_data.tolist()
            return channel_data
        else:
            # Multi-channel audio - assume interleaved
            num_channels = len(channels)
            if len(float_array) % num_channels != 0:
                # Truncate to complete samples
                truncate_length = (len(float_array) // num_channels) * num_channels
                float_array = float_array[:truncate_length]
            
            # Reshape to separate channels
            reshaped = float_array.reshape(-1, num_channels)
            
            channel_data = {}
            for ch_id, config in channels.items():
                if config.enabled and ch_id < num_channels:
                    # Extract channel data and scale to voltage
                    channel_samples = reshaped[:, ch_id]
                    voltage_data = channel_samples * (config.voltage_range / 2.0)
                    channel_data[ch_id] = voltage_data.tolist()
            
            return channel_data


class ASCIICSVParser(ADCDataParser):
    """Parser for ASCII CSV data."""
    
    async def parse(self, raw_data: bytes, channels: Dict[int, ADCChannelConfig]) -> Dict[int, List[float]]:
        """Parse ASCII CSV data."""
        try:
            text_data = raw_data.decode('utf-8')
            lines = text_data.strip().split('\n')
            
            channel_data = {ch_id: [] for ch_id, config in channels.items() if config.enabled}
            
            for line in lines:
                if not line.strip():
                    continue
                    
                values = line.split(',')
                for ch_id, config in channels.items():
                    if not config.enabled or ch_id >= len(values):
                        continue
                    
                    try:
                        raw_value = float(values[ch_id])
                        voltage = config.raw_to_voltage(raw_value)
                        channel_data[ch_id].append(voltage)
                    except (ValueError, IndexError):
                        continue
            
            return channel_data
            
        except UnicodeDecodeError:
            logger.error("Failed to decode ASCII CSV data")
            return {}


class JSONArrayParser(ADCDataParser):
    """Parser for JSON array data."""
    
    async def parse(self, raw_data: bytes, channels: Dict[int, ADCChannelConfig]) -> Dict[int, List[float]]:
        """Parse JSON array data."""
        try:
            text_data = raw_data.decode('utf-8')
            json_data = json.loads(text_data)
            
            channel_data = {}
            
            if isinstance(json_data, dict):
                # Format: {"channel_0": [values], "channel_1": [values]}
                for ch_id, config in channels.items():
                    if not config.enabled:
                        continue
                    
                    key = f"channel_{ch_id}"
                    if key in json_data:
                        raw_values = json_data[key]
                        voltage_values = [config.raw_to_voltage(val) for val in raw_values]
                        channel_data[ch_id] = voltage_values
                        
            elif isinstance(json_data, list):
                # Format: [[ch0_val, ch1_val], [ch0_val, ch1_val], ...]
                channel_data = {ch_id: [] for ch_id, config in channels.items() if config.enabled}
                
                for sample in json_data:
                    if isinstance(sample, list):
                        for ch_id, config in channels.items():
                            if config.enabled and ch_id < len(sample):
                                voltage = config.raw_to_voltage(sample[ch_id])
                                channel_data[ch_id].append(voltage)
            
            return channel_data
            
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error("Failed to parse JSON data", error=str(e))
            return {}


class UniversalADCInterface:
    """Universal interface for any ADC hardware."""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.channels = {}
        self.is_active = False
        self.sample_rate = 1000.0  # Default 1 kHz
        self.data_format = ADCDataFormat.RAW_BINARY
        self.parser = None
        self.custom_parser_func = None
        
        # Data callback
        self.data_callback = None
        
        # Statistics
        self.total_samples = 0
        self.total_bytes = 0
        self.last_sample_time = 0
        
    def add_channel(
        self,
        channel_id: int,
        voltage_range: float = 3.3,
        resolution_bits: int = 12,
        **kwargs
    ) -> ADCChannelConfig:
        """Add a channel configuration."""
        config = ADCChannelConfig(
            channel_id=channel_id,
            voltage_range=voltage_range,
            resolution_bits=resolution_bits,
            **kwargs
        )
        self.channels[channel_id] = config
        logger.info("ADC channel added", device_id=self.device_id, channel_id=channel_id, config=config.to_dict())
        return config
    
    def set_data_format(
        self,
        data_format: ADCDataFormat,
        parser_params: Optional[Dict[str, Any]] = None,
        custom_parser: Optional[Callable] = None
    ):
        """Set the data format and create appropriate parser."""
        self.data_format = data_format
        
        if data_format == ADCDataFormat.RAW_BINARY:
            params = parser_params or {}
            self.parser = RawBinaryParser(
                bytes_per_sample=params.get("bytes_per_sample", 2),
                num_channels=params.get("num_channels", len(self.channels)),
                byte_order=params.get("byte_order", "little")
            )
        elif data_format == ADCDataFormat.AUDIO_FLOAT32:
            self.parser = AudioFloat32Parser()
        elif data_format == ADCDataFormat.ASCII_CSV:
            self.parser = ASCIICSVParser()
        elif data_format == ADCDataFormat.JSON_ARRAY:
            self.parser = JSONArrayParser()
        elif data_format == ADCDataFormat.CUSTOM:
            if custom_parser:
                self.custom_parser_func = custom_parser
            else:
                raise ValueError("Custom parser function required for CUSTOM format")
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        
        logger.info("ADC data format set", device_id=self.device_id, format=data_format.value)
    
    def set_data_callback(self, callback: Callable[[Dict[int, List[float]], float], None]):
        """Set callback function for processed data."""
        self.data_callback = callback
    
    async def process_raw_data(self, raw_data: bytes) -> Optional[Dict[int, List[float]]]:
        """Process raw data through the configured parser."""
        if not raw_data:
            return None
        
        try:
            # Update statistics
            self.total_bytes += len(raw_data)
            self.last_sample_time = time.time()
            
            # Parse data
            if self.data_format == ADCDataFormat.CUSTOM and self.custom_parser_func:
                channel_data = await self.custom_parser_func(raw_data, self.channels)
            elif self.parser:
                channel_data = await self.parser.parse(raw_data, self.channels)
            else:
                logger.error("No parser configured for data format", format=self.data_format)
                return None
            
            # Update sample count
            if channel_data:
                max_samples = max(len(samples) for samples in channel_data.values())
                self.total_samples += max_samples
            
            # Call data callback if set
            if self.data_callback and channel_data:
                try:
                    await self.data_callback(channel_data, self.sample_rate)
                except Exception as e:
                    logger.error("Data callback failed", error=str(e))
            
            return channel_data
            
        except Exception as e:
            logger.error("Failed to process raw data", device_id=self.device_id, error=str(e))
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get interface statistics."""
        current_time = time.time()
        uptime = current_time - (self.last_sample_time - self.total_samples / self.sample_rate) if self.total_samples > 0 else 0
        
        return {
            "device_id": self.device_id,
            "is_active": self.is_active,
            "total_samples": self.total_samples,
            "total_bytes": self.total_bytes,
            "sample_rate": self.sample_rate,
            "channels": {ch_id: config.to_dict() for ch_id, config in self.channels.items()},
            "data_format": self.data_format.value,
            "last_sample_time": self.last_sample_time,
            "uptime_seconds": uptime,
            "avg_bytes_per_second": self.total_bytes / uptime if uptime > 0 else 0
        }
    
    async def start(self):
        """Start the ADC interface."""
        self.is_active = True
        logger.info("ADC interface started", device_id=self.device_id)
    
    async def stop(self):
        """Stop the ADC interface."""
        self.is_active = False
        logger.info("ADC interface stopped", device_id=self.device_id)


class SerialADCInterface(UniversalADCInterface):
    """ADC interface for serial-connected devices."""
    
    def __init__(self, device_id: str, port: str, baudrate: int = 115200):
        super().__init__(device_id)
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.read_task = None
        
        if not HAS_SERIAL:
            raise ImportError("pyserial is required for SerialADCInterface")
    
    async def start(self):
        """Start serial communication."""
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            self.read_task = asyncio.create_task(self._read_loop())
            await super().start()
            logger.info("Serial ADC started", device_id=self.device_id, port=self.port, baudrate=self.baudrate)
        except Exception as e:
            logger.error("Failed to start serial ADC", device_id=self.device_id, error=str(e))
            raise
    
    async def stop(self):
        """Stop serial communication."""
        await super().stop()
        
        if self.read_task:
            self.read_task.cancel()
            try:
                await self.read_task
            except asyncio.CancelledError:
                pass
        
        if self.serial_connection:
            self.serial_connection.close()
            self.serial_connection = None
        
        logger.info("Serial ADC stopped", device_id=self.device_id)
    
    async def _read_loop(self):
        """Background task to read from serial port."""
        while self.is_active and self.serial_connection:
            try:
                if self.serial_connection.in_waiting > 0:
                    data = self.serial_connection.read(self.serial_connection.in_waiting)
                    if data:
                        await self.process_raw_data(data)
                else:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                    
            except Exception as e:
                logger.error("Serial read error", device_id=self.device_id, error=str(e))
                await asyncio.sleep(0.1)


class AudioADCInterface(UniversalADCInterface):
    """ADC interface for audio devices (microphones, line inputs)."""
    
    def __init__(self, device_id: str, device_index: Optional[int] = None):
        super().__init__(device_id)
        self.device_index = device_index
        self.audio_stream = None
        
        if not HAS_AUDIO:
            raise ImportError("sounddevice is required for AudioADCInterface")
        
        # Audio-specific settings
        self.sample_rate = 44100.0
        self.channels_count = 1
        self.dtype = np.float32
        self.blocksize = 1024
        
        # Set audio format
        self.set_data_format(ADCDataFormat.AUDIO_FLOAT32)
    
    def audio_callback(self, indata, frames, time, status):
        """Audio input callback."""
        if status:
            logger.warning("Audio status", device_id=self.device_id, status=status)
        
        # Convert to bytes for processing
        audio_bytes = indata.tobytes()
        
        # Process asynchronously
        asyncio.create_task(self.process_raw_data(audio_bytes))
    
    async def start(self):
        """Start audio capture."""
        try:
            self.audio_stream = sd.InputStream(
                device=self.device_index,
                channels=self.channels_count,
                samplerate=self.sample_rate,
                dtype=self.dtype,
                blocksize=self.blocksize,
                callback=self.audio_callback
            )
            
            self.audio_stream.start()
            await super().start()
            
            logger.info("Audio ADC started", 
                       device_id=self.device_id,
                       device_index=self.device_index,
                       sample_rate=self.sample_rate,
                       channels=self.channels_count)
                       
        except Exception as e:
            logger.error("Failed to start audio ADC", device_id=self.device_id, error=str(e))
            raise
    
    async def stop(self):
        """Stop audio capture."""
        await super().stop()
        
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        
        logger.info("Audio ADC stopped", device_id=self.device_id)


# Factory function for creating ADC interfaces
def create_adc_interface(
    interface_type: str,
    device_id: str,
    **kwargs
) -> UniversalADCInterface:
    """Factory function to create appropriate ADC interface."""
    
    if interface_type == "serial":
        return SerialADCInterface(device_id, **kwargs)
    elif interface_type == "audio":
        return AudioADCInterface(device_id, **kwargs)
    elif interface_type == "universal":
        return UniversalADCInterface(device_id)
    else:
        raise ValueError(f"Unknown interface type: {interface_type}")


# Example usage functions
async def example_microphone_adc():
    """Example: Set up microphone as ADC source."""
    
    # Create audio ADC interface
    adc = AudioADCInterface("microphone_adc")
    
    # Add single channel for mono audio
    adc.add_channel(0, voltage_range=2.0, resolution_bits=16, coupling="AC")
    
    # Set data callback
    async def data_handler(channel_data, sample_rate):
        print(f"Received {len(channel_data[0])} samples at {sample_rate} Hz")
    
    adc.set_data_callback(data_handler)
    
    # Start capture
    await adc.start()
    
    # Let it run for 10 seconds
    await asyncio.sleep(10)
    
    # Stop
    await adc.stop()
    
    # Print statistics
    stats = adc.get_statistics()
    print(f"Captured {stats['total_samples']} samples, {stats['total_bytes']} bytes")


async def example_serial_adc():
    """Example: Set up serial ADC."""
    
    # Create serial ADC interface
    adc = SerialADCInterface("serial_adc", port="/dev/ttyUSB0", baudrate=115200)
    
    # Add channels
    adc.add_channel(0, voltage_range=3.3, resolution_bits=12)
    adc.add_channel(1, voltage_range=5.0, resolution_bits=12)
    
    # Set binary data format
    adc.set_data_format(
        ADCDataFormat.RAW_BINARY,
        parser_params={
            "bytes_per_sample": 2,
            "num_channels": 2,
            "byte_order": "little"
        }
    )
    
    # Set data callback
    async def data_handler(channel_data, sample_rate):
        for ch_id, data in channel_data.items():
            print(f"Channel {ch_id}: {len(data)} samples, latest = {data[-1]:.3f}V")
    
    adc.set_data_callback(data_handler)
    
    # Start
    await adc.start()
    
    # Run until interrupted
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    
    # Stop
    await adc.stop()


if __name__ == "__main__":
    # Run microphone example
    asyncio.run(example_microphone_adc())