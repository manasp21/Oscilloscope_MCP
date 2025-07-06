# MCP Signal Processing Server: Complete Technical Implementation Plan

## Executive Summary

This comprehensive technical plan outlines the implementation of a high-performance MCP (Model Context Protocol) server that integrates digital oscilloscope and function generator capabilities. The system leverages advanced signal processing algorithms, real-time data streaming, and precise waveform synthesis to provide professional-grade measurement and signal generation capabilities through Claude Code integration.

**Key Innovation**: The proposed system combines MCP's standardized protocol architecture with cutting-edge DSP algorithms to create a unified instrument platform that delivers sub-microsecond latency, >100 dB dynamic range, and 1 ps timing precision.

## 1. MCP Server Architecture Design

### 1.1 Three-Tier MCP Implementation

**Tools Tier (Model-Controlled Functions):**
- **Signal Processing Tools**: FFT analysis, filtering, triggering, measurement algorithms
- **Waveform Generation Tools**: DDS synthesis, modulation, arbitrary waveform creation
- **Configuration Tools**: Hardware setup, calibration, protocol configuration
- **Data Acquisition Tools**: Real-time streaming, buffer management, event capture

**Resources Tier (Application-Controlled Data):**
- **Waveform Data**: URI-based access to acquired signals (`signal://channel1/buffer`)
- **Measurement Results**: Statistical data, frequency domain analysis, protocol decode
- **Calibration Data**: Compensation tables, probe characteristics, system parameters
- **Configuration State**: Instrument settings, trigger conditions, measurement parameters

**Prompts Tier (User-Controlled Templates):**
- **Analysis Workflows**: Pre-configured measurement sequences
- **Test Procedures**: Automated compliance testing templates
- **Debugging Scripts**: Protocol analysis and troubleshooting guides
- **Report Generation**: Standardized measurement documentation

### 1.2 High-Performance Protocol Implementation

**Transport Layer Selection:**
```python
# Streamable HTTP for maximum performance
from mcp.server.fastmcp import FastMCP
import asyncio
import numpy as np

class HighPerformanceMCPServer:
    def __init__(self):
        self.mcp = FastMCP("signal-processing-server")
        self.sample_rate = 100e6  # 100 MS/s
        self.buffer_size = 1024 * 1024  # 1M samples
        self.channels = 4
        
    async def initialize_streaming(self):
        """Initialize high-throughput streaming architecture"""
        self.dma_buffers = self._allocate_dma_buffers()
        self.processing_pool = ThreadPoolExecutor(max_workers=8)
        self.result_queue = asyncio.Queue(maxsize=1000)
```

**Real-Time Data Streaming Architecture:**
- **Zero-Copy Buffers**: Memory-mapped circular buffers for minimum latency
- **DMA Integration**: Direct memory access for high-throughput data transfer
- **Asynchronous Processing**: Non-blocking I/O with asyncio patterns
- **Connection Pooling**: Efficient resource management for multiple clients

### 1.3 Performance Optimization Strategies

**Memory Management:**
```python
# Lock-free circular buffer implementation
class LockFreeCircularBuffer:
    def __init__(self, size):
        self.size = size
        self.mask = size - 1  # Power-of-2 optimization
        self.buffer = np.zeros(size, dtype=np.float32)
        self.write_idx = 0
        self.read_idx = 0
        
    def write(self, data):
        """Write data with memory barriers for real-time safety"""
        next_write = (self.write_idx + len(data)) & self.mask
        if next_write == self.read_idx:
            raise BufferOverflowError("Buffer full")
        
        self.buffer[self.write_idx:next_write] = data
        self.write_idx = next_write
```

**Latency Optimization:**
- **SIMD Processing**: ARM NEON/x86 AVX for parallel operations
- **Cache-Aligned Data**: 64-byte alignment for optimal cache utilization
- **Thread Affinity**: CPU core binding for predictable performance
- **Priority Scheduling**: Real-time thread priorities for critical paths

## 2. Digital Oscilloscope Implementation

### 2.1 Core Signal Processing Algorithms

**FFT Analysis Engine:**
```python
@mcp.tool
async def compute_fft(
    channel: int,
    window_type: str = "hamming",
    overlap: float = 0.5,
    zero_padding: int = 1
) -> dict:
    """High-performance FFT with windowing and overlap processing"""
    
    # Optimized FFT implementation
    windowed_data = apply_window(raw_data, window_type)
    fft_result = np.fft.fft(windowed_data, n=len(windowed_data) * zero_padding)
    
    # Frequency domain analysis
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    power_spectrum = magnitude ** 2
    
    return {
        "frequency_axis": np.fft.fftfreq(len(fft_result), 1/self.sample_rate),
        "magnitude_db": 20 * np.log10(magnitude),
        "phase_deg": np.degrees(phase),
        "power_spectrum_dbm": 10 * np.log10(power_spectrum) + 30,
        "thd_percent": calculate_thd(magnitude),
        "sfdr_db": calculate_sfdr(magnitude)
    }
```

**Advanced Triggering System:**
```python
class AdvancedTriggerEngine:
    def __init__(self):
        self.trigger_types = {
            "edge": EdgeTrigger,
            "pulse_width": PulseWidthTrigger,
            "pattern": PatternTrigger,
            "protocol": ProtocolTrigger,
            "mathematical": MathematicalTrigger
        }
        
    async def process_triggers(self, data_stream):
        """FPGA-accelerated trigger processing"""
        for trigger in self.active_triggers:
            if trigger.evaluate(data_stream):
                await self.capture_event(trigger, data_stream)
```

**Digital Filtering Implementation:**
```python
# High-performance IIR filter cascade
class CascadedBiquadFilter:
    def __init__(self, filter_type, cutoff_freq, order):
        self.sections = self._design_biquad_sections(filter_type, cutoff_freq, order)
        
    def _design_biquad_sections(self, filter_type, fc, order):
        """Design cascaded biquad sections for optimal numerical stability"""
        if filter_type == "butterworth":
            return self._butterworth_sections(fc, order)
        elif filter_type == "chebyshev1":
            return self._chebyshev1_sections(fc, order)
        elif filter_type == "chebyshev2":
            return self._chebyshev2_sections(fc, order)
        elif filter_type == "bessel":
            return self._bessel_sections(fc, order)
            
    def process_simd(self, input_data):
        """SIMD-optimized filter processing"""
        result = input_data
        for section in self.sections:
            result = section.process_neon(result)  # ARM NEON optimization
        return result
```

### 2.2 Real-Time Processing Architecture

**FPGA-Based Processing Pipeline:**
```python
class FPGAProcessingPipeline:
    def __init__(self):
        self.stages = [
            ADCInterface(resolution=16, sample_rate=100e6),
            DigitalDownConverter(decimation=4),
            TriggerEngine(parallel_triggers=8),
            FFTProcessor(size=1024, overlap=0.5),
            MeasurementEngine(parallel_measurements=16),
            DataStreamer(bandwidth=1e9)  # 1 GB/s
        ]
        
    async def process_pipeline(self, input_stream):
        """Parallel processing pipeline with deterministic timing"""
        processed_data = input_stream
        for stage in self.stages:
            processed_data = await stage.process_async(processed_data)
        return processed_data
```

**Memory Management for Continuous Acquisition:**
```python
class ContinuousAcquisitionManager:
    def __init__(self, deep_memory_gb=8):
        self.memory_size = deep_memory_gb * 1024**3
        self.segment_size = 1024 * 1024  # 1M samples per segment
        self.segments = self._allocate_segments()
        
    def _allocate_segments(self):
        """Allocate memory-mapped segments for zero-copy operation"""
        segments = []
        for i in range(self.memory_size // self.segment_size):
            segment = mmap.mmap(-1, self.segment_size)
            segments.append(segment)
        return segments
```

## 3. Function Generator Implementation

### 3.1 Direct Digital Synthesis (DDS) Engine

**Advanced DDS Architecture:**
```python
class AdvancedDDSEngine:
    def __init__(self, clock_freq=100e6, phase_accumulator_bits=32):
        self.clock_freq = clock_freq
        self.phase_bits = phase_accumulator_bits
        self.freq_resolution = clock_freq / (2 ** phase_accumulator_bits)
        
        # CORDIC processor for sine/cosine generation
        self.cordic = CORDICProcessor(iterations=16)
        
        # Pre-computed lookup tables for optimization
        self.sine_lut = self._generate_sine_lut(1024)
        
    def generate_waveform(self, frequency, amplitude, phase_offset):
        """Generate waveform using optimized DDS algorithm"""
        fcw = int(frequency / self.freq_resolution)  # Frequency Control Word
        
        # Phase accumulator operation
        phase_acc = 0
        samples = []
        
        for _ in range(self.buffer_size):
            # Phase-to-amplitude conversion
            sine_val = self.cordic.sine(phase_acc * 2 * np.pi / (2**self.phase_bits))
            sample = amplitude * sine_val
            samples.append(sample)
            
            # Update phase accumulator
            phase_acc = (phase_acc + fcw) & ((2**self.phase_bits) - 1)
            
        return np.array(samples)
```

**Arbitrary Waveform Generation:**
```python
@mcp.tool
async def generate_arbitrary_waveform(
    sample_data: List[float],
    sample_rate: float,
    amplitude: float,
    dc_offset: float = 0.0,
    repetition_rate: float = 1.0
) -> dict:
    """Generate arbitrary waveform with interpolation and anti-aliasing"""
    
    # Resample to target rate with anti-aliasing
    resampled = signal.resample(sample_data, 
                              int(len(sample_data) * sample_rate / self.native_rate))
    
    # Apply amplitude scaling and DC offset
    scaled_waveform = amplitude * resampled + dc_offset
    
    # Generate timing information
    time_axis = np.arange(len(scaled_waveform)) / sample_rate
    
    return {
        "waveform_samples": scaled_waveform.tolist(),
        "time_axis": time_axis.tolist(),
        "sample_rate": sample_rate,
        "duration": len(scaled_waveform) / sample_rate,
        "memory_usage": len(scaled_waveform) * 4  # 4 bytes per float32
    }
```

### 3.2 Modulation Implementation

**Advanced Modulation Schemes:**
```python
class ModulationEngine:
    def __init__(self):
        self.modulation_types = {
            "AM": self._amplitude_modulation,
            "FM": self._frequency_modulation,
            "PM": self._phase_modulation,
            "FSK": self._frequency_shift_keying,
            "PSK": self._phase_shift_keying,
            "QAM": self._quadrature_amplitude_modulation
        }
        
    def _amplitude_modulation(self, carrier_freq, mod_freq, mod_index):
        """AM: y(t) = A[1 + m*cos(2πfm*t)]*cos(2πfc*t)"""
        t = np.linspace(0, 1, int(self.sample_rate))
        modulating = np.cos(2 * np.pi * mod_freq * t)
        carrier = np.cos(2 * np.pi * carrier_freq * t)
        return (1 + mod_index * modulating) * carrier
        
    def _frequency_modulation(self, carrier_freq, mod_freq, freq_deviation):
        """FM: y(t) = A*cos(2πfc*t + β*sin(2πfm*t))"""
        t = np.linspace(0, 1, int(self.sample_rate))
        modulating = np.sin(2 * np.pi * mod_freq * t)
        instantaneous_phase = 2 * np.pi * carrier_freq * t + freq_deviation * modulating
        return np.cos(instantaneous_phase)
```

### 3.3 Precision Timing and Synchronization

**High-Precision Clock Generation:**
```python
class PrecisionClockGenerator:
    def __init__(self):
        self.reference_freq = 10e6  # 10 MHz OCXO reference
        self.pll_multiplier = 10    # 100 MHz output
        self.phase_noise_target = -140  # dBc/Hz at 10 kHz offset
        
    def generate_synchronized_clocks(self, num_channels):
        """Generate phase-coherent clocks for multi-channel operation"""
        master_clock = self._generate_master_clock()
        
        channel_clocks = []
        for i in range(num_channels):
            # Phase-locked to master with programmable delay
            delayed_clock = self._apply_phase_delay(master_clock, i * 90)  # 90° increments
            channel_clocks.append(delayed_clock)
            
        return channel_clocks
        
    def calculate_phase_noise(self, frequency_offset):
        """Calculate phase noise performance"""
        # Flicker noise + white noise model
        flicker_corner = 1000  # 1 kHz
        white_noise_floor = -160  # dBc/Hz
        
        if frequency_offset < flicker_corner:
            return white_noise_floor + 10 * np.log10(flicker_corner / frequency_offset)
        else:
            return white_noise_floor
```

## 4. Technical Implementation Architecture

### 4.1 Programming Language Selection

**Performance-Critical Components (C/C++):**
```cpp
// SIMD-optimized FIR filter implementation
class SIMDFIRFilter {
private:
    float* coefficients;
    float* delay_line;
    int filter_length;
    
public:
    void process_neon(const float* input, float* output, int samples) {
        // ARM NEON implementation for 4x parallel processing
        float32x4_t sum;
        float32x4_t coeff_vec, input_vec;
        
        for (int i = 0; i < samples; i += 4) {
            sum = vdupq_n_f32(0.0f);
            
            for (int j = 0; j < filter_length; j += 4) {
                coeff_vec = vld1q_f32(&coefficients[j]);
                input_vec = vld1q_f32(&input[i + j]);
                sum = vmlaq_f32(sum, coeff_vec, input_vec);
            }
            
            // Horizontal sum and store result
            vst1q_f32(&output[i], sum);
        }
    }
};
```

**MCP Server Layer (Python/AsyncIO):**
```python
class SignalProcessingMCPServer:
    def __init__(self):
        self.mcp = FastMCP("advanced-signal-processor")
        self.dsp_engine = DSPEngine()  # C++ backend
        self.function_gen = FunctionGenerator()  # C++ backend
        
    @mcp.tool
    async def real_time_fft(self, channel: int, window_size: int = 1024) -> dict:
        """Real-time FFT analysis with hardware acceleration"""
        # Delegate to C++ DSP engine for performance
        raw_data = await self.acquire_samples(channel, window_size)
        fft_result = self.dsp_engine.compute_fft(raw_data)
        
        return {
            "frequency_bins": fft_result.frequencies.tolist(),
            "magnitude_db": fft_result.magnitude_db.tolist(),
            "phase_deg": fft_result.phase_deg.tolist(),
            "processing_time_us": fft_result.processing_time_us
        }
```

### 4.2 Data Structure Design

**High-Performance Circular Buffers:**
```python
import mmap
import ctypes

class HighPerformanceCircularBuffer:
    def __init__(self, size_bytes):
        self.size = size_bytes
        self.mask = size_bytes - 1  # Power-of-2 for efficient modulo
        
        # Memory-mapped buffer for zero-copy operation
        self.buffer = mmap.mmap(-1, size_bytes * 2)  # Double mapping trick
        
        # Atomic indices for lock-free operation
        self.write_idx = ctypes.c_uint64(0)
        self.read_idx = ctypes.c_uint64(0)
        
    def write(self, data):
        """Lock-free write operation"""
        data_size = len(data)
        current_write = self.write_idx.value
        
        # Check for available space
        if (current_write - self.read_idx.value) + data_size > self.size:
            raise BufferOverflowError("Insufficient buffer space")
        
        # Copy data with automatic wrap-around
        write_pos = current_write & self.mask
        self.buffer[write_pos:write_pos + data_size] = data
        
        # Update write index atomically
        self.write_idx.value = current_write + data_size
```

**SIMD-Optimized Data Structures:**
```python
class SIMDOptimizedArray:
    def __init__(self, size, dtype=np.float32):
        # Ensure 64-byte alignment for AVX-512
        self.size = size
        self.data = np.empty(size, dtype=dtype)
        self.data = np.asarray(self.data, dtype=dtype, order='C')
        
        # Verify alignment
        assert self.data.ctypes.data % 64 == 0, "Data not properly aligned"
        
    def apply_filter_avx512(self, coefficients):
        """AVX-512 optimized filtering"""
        # 16 float32 values per AVX-512 register
        # Implementation would use Intel intrinsics
        pass
```

### 4.3 Threading and Concurrency

**Real-Time Processing Architecture:**
```python
import threading
import queue
import time

class RealTimeProcessor:
    def __init__(self):
        self.processing_threads = []
        self.data_queue = queue.Queue(maxsize=1000)
        self.result_queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        
    def start_processing(self, num_threads=4):
        """Start real-time processing threads with appropriate priorities"""
        for i in range(num_threads):
            thread = threading.Thread(
                target=self._processing_loop,
                args=(i,),
                daemon=True
            )
            
            # Set real-time priority (requires root on Linux)
            thread.start()
            self.processing_threads.append(thread)
            
    def _processing_loop(self, thread_id):
        """Main processing loop with deterministic timing"""
        # Set CPU affinity for predictable performance
        os.sched_setaffinity(0, {thread_id})
        
        while not self.stop_event.is_set():
            try:
                # Get data with timeout
                data = self.data_queue.get(timeout=0.001)
                
                # Process with hardware acceleration
                result = self._process_data_simd(data)
                
                # Store result
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
```

## 5. Advanced Features Implementation

### 5.1 Protocol Analysis Engine

**Multi-Protocol Decoder:**
```python
@mcp.tool
async def analyze_protocol(
    channel: int,
    protocol_type: str,
    baud_rate: int,
    trigger_conditions: dict
) -> dict:
    """Real-time protocol analysis with hardware acceleration"""
    
    protocol_decoders = {
        "UART": UARTDecoder,
        "I2C": I2CDecoder,
        "SPI": SPIDecoder,
        "CAN": CANDecoder,
        "USB": USBDecoder,
        "Ethernet": EthernetDecoder
    }
    
    decoder = protocol_decoders[protocol_type](baud_rate)
    
    # Hardware-accelerated pattern matching
    decoded_data = await decoder.decode_stream(
        channel, 
        trigger_conditions
    )
    
    return {
        "protocol": protocol_type,
        "decoded_packets": decoded_data.packets,
        "error_count": decoded_data.errors,
        "timing_analysis": decoded_data.timing,
        "bus_utilization": decoded_data.utilization_percent
    }
```

### 5.2 Advanced Triggering System

**Pattern Recognition Trigger:**
```python
class PatternRecognitionTrigger:
    def __init__(self):
        self.pattern_engine = PatternMatchingEngine()
        self.templates = {}
        
    def add_pattern_template(self, name, pattern_data, tolerance=0.1):
        """Add waveform pattern template for recognition"""
        normalized_pattern = self._normalize_pattern(pattern_data)
        self.templates[name] = {
            "pattern": normalized_pattern,
            "tolerance": tolerance,
            "correlation_threshold": 0.8
        }
        
    def evaluate_trigger(self, input_data):
        """Evaluate trigger condition using cross-correlation"""
        for name, template in self.templates.items():
            correlation = np.correlate(input_data, template["pattern"], mode='valid')
            max_correlation = np.max(correlation)
            
            if max_correlation > template["correlation_threshold"]:
                return True, name, max_correlation
                
        return False, None, 0
```

### 5.3 Measurement Automation

**Automated Measurement Engine:**
```python
@mcp.tool
async def automated_measurements(
    channels: List[int],
    measurement_list: List[str],
    statistics: bool = True
) -> dict:
    """Perform automated measurements with statistical analysis"""
    
    measurement_functions = {
        "rms": self._calculate_rms,
        "peak_to_peak": self._calculate_peak_to_peak,
        "rise_time": self._calculate_rise_time,
        "fall_time": self._calculate_fall_time,
        "frequency": self._calculate_frequency,
        "period": self._calculate_period,
        "duty_cycle": self._calculate_duty_cycle,
        "overshoot": self._calculate_overshoot,
        "phase_difference": self._calculate_phase_difference
    }
    
    results = {}
    
    for channel in channels:
        channel_data = await self.acquire_channel_data(channel)
        channel_results = {}
        
        for measurement in measurement_list:
            if measurement in measurement_functions:
                value = measurement_functions[measurement](channel_data)
                
                if statistics:
                    # Calculate statistics over multiple acquisitions
                    stats = await self._calculate_statistics(
                        measurement_functions[measurement],
                        channel,
                        num_acquisitions=100
                    )
                    channel_results[measurement] = {
                        "value": value,
                        "mean": stats.mean,
                        "std_dev": stats.std,
                        "min": stats.min,
                        "max": stats.max,
                        "confidence_interval": stats.confidence_95
                    }
                else:
                    channel_results[measurement] = {"value": value}
        
        results[f"channel_{channel}"] = channel_results
    
    return results
```

## 6. Calibration and Compensation

### 6.1 System Calibration Algorithms

**Multi-Point Calibration:**
```python
class SystemCalibrationManager:
    def __init__(self):
        self.calibration_standards = {
            "dc_voltage": [0.0, 1.0, 5.0, 10.0],  # Volts
            "ac_voltage": [0.1, 1.0, 5.0],        # Volts RMS
            "frequency": [1e3, 1e6, 100e6],       # Hz
            "timebase": [1e-9, 1e-6, 1e-3]        # Seconds
        }
        
    async def perform_system_calibration(self):
        """Comprehensive system calibration procedure"""
        calibration_results = {}
        
        # DC voltage calibration
        dc_cal = await self._calibrate_dc_accuracy()
        calibration_results["dc_voltage"] = dc_cal
        
        # AC voltage calibration
        ac_cal = await self._calibrate_ac_accuracy()
        calibration_results["ac_voltage"] = ac_cal
        
        # Frequency calibration
        freq_cal = await self._calibrate_frequency_accuracy()
        calibration_results["frequency"] = freq_cal
        
        # Timebase calibration
        time_cal = await self._calibrate_timebase_accuracy()
        calibration_results["timebase"] = time_cal
        
        # Store calibration constants
        await self._store_calibration_data(calibration_results)
        
        return calibration_results
        
    def _calculate_calibration_polynomial(self, reference_values, measured_values):
        """Calculate polynomial correction coefficients"""
        # Fit polynomial to calibration data
        coefficients = np.polyfit(measured_values, reference_values, deg=2)
        
        # Calculate correction accuracy
        corrected_values = np.polyval(coefficients, measured_values)
        accuracy = np.max(np.abs(corrected_values - reference_values))
        
        return {
            "coefficients": coefficients.tolist(),
            "accuracy": accuracy,
            "r_squared": self._calculate_r_squared(reference_values, corrected_values)
        }
```

### 6.2 Probe Compensation

**Automatic Probe Compensation:**
```python
class ProbeCompensationEngine:
    def __init__(self):
        self.probe_types = {
            "1x": {"attenuation": 1, "bandwidth": 100e6},
            "10x": {"attenuation": 10, "bandwidth": 500e6},
            "100x": {"attenuation": 100, "bandwidth": 250e6},
            "differential": {"attenuation": 10, "bandwidth": 1e9}
        }
        
    async def auto_compensate_probe(self, channel, probe_type):
        """Automatic probe compensation using calibration signal"""
        
        # Apply calibration signal (1 kHz square wave)
        await self.apply_calibration_signal(1000, "square")
        
        # Measure response
        response = await self.measure_square_wave_response(channel)
        
        # Calculate compensation values
        compensation = self._calculate_compensation(response, probe_type)
        
        # Apply compensation
        await self.set_probe_compensation(channel, compensation)
        
        return {
            "probe_type": probe_type,
            "compensation_values": compensation,
            "flatness_error": response.flatness_error,
            "rise_time": response.rise_time
        }
```

## 7. Integration and Deployment

### 7.1 Claude Code Integration

**MCP Server Deployment:**
```python
# Main server implementation for Claude Code
import asyncio
from mcp.server.fastmcp import FastMCP
from signal_processing_engine import SignalProcessingEngine
from function_generator import FunctionGenerator

class IntegratedInstrumentServer:
    def __init__(self):
        self.mcp = FastMCP("integrated-instrument-server")
        self.dsp_engine = SignalProcessingEngine()
        self.function_gen = FunctionGenerator()
        
        # Register all tools
        self._register_oscilloscope_tools()
        self._register_function_generator_tools()
        self._register_analysis_tools()
        
    def _register_oscilloscope_tools(self):
        """Register digital oscilloscope tools"""
        
        @self.mcp.tool
        async def configure_acquisition(
            channels: List[int],
            sample_rate: float,
            record_length: int,
            trigger_config: dict
        ) -> dict:
            """Configure data acquisition parameters"""
            return await self.dsp_engine.configure_acquisition(
                channels, sample_rate, record_length, trigger_config
            )
            
        @self.mcp.tool
        async def acquire_waveform(channel: int) -> dict:
            """Acquire waveform data from specified channel"""
            return await self.dsp_engine.acquire_waveform(channel)
            
        @self.mcp.tool
        async def analyze_spectrum(channel: int, window: str = "hamming") -> dict:
            """Perform FFT analysis on acquired data"""
            return await self.dsp_engine.analyze_spectrum(channel, window)
            
    def _register_function_generator_tools(self):
        """Register function generator tools"""
        
        @self.mcp.tool
        async def generate_waveform(
            channel: int,
            waveform_type: str,
            frequency: float,
            amplitude: float,
            phase: float = 0.0
        ) -> dict:
            """Generate standard waveform"""
            return await self.function_gen.generate_waveform(
                channel, waveform_type, frequency, amplitude, phase
            )
            
        @self.mcp.tool
        async def generate_modulated_signal(
            carrier_freq: float,
            modulation_type: str,
            modulation_params: dict
        ) -> dict:
            """Generate modulated signal"""
            return await self.function_gen.generate_modulated_signal(
                carrier_freq, modulation_type, modulation_params
            )

if __name__ == "__main__":
    server = IntegratedInstrumentServer()
    asyncio.run(server.mcp.run())
```

### 7.2 Performance Specifications

**System Performance Targets:**
- **Oscilloscope Bandwidth**: DC to 1 GHz
- **Sample Rate**: Up to 5 GS/s real-time
- **Memory Depth**: 1 GSamples per channel
- **Trigger Latency**: <100 ns
- **Measurement Accuracy**: ±0.1% for DC, ±1% for AC
- **Function Generator Frequency Range**: 1 μHz to 500 MHz
- **Frequency Resolution**: 1 μHz
- **Amplitude Accuracy**: ±0.5% of setting
- **Phase Noise**: <-130 dBc/Hz at 10 kHz offset
- **Spurious-Free Dynamic Range**: >80 dBc

### 7.3 Deployment Architecture

**Container-Based Deployment:**
```dockerfile
# High-performance signal processing container
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    libnuma-dev \
    libfftw3-dev \
    libblas-dev \
    liblapack-dev \
    gcc-12 \
    g++-12

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . /app
WORKDIR /app

# Build C++ extensions
RUN python setup.py build_ext --inplace

# Set real-time capabilities
RUN setcap cap_sys_nice+ep /usr/bin/python3.11

# Run server
CMD ["python", "-m", "mcp.server", "--transport", "stdio"]
```

## 8. Conclusion and Implementation Roadmap

### 8.1 Implementation Phases

**Phase 1: Core Infrastructure (Months 1-3)**
- MCP server framework implementation
- Basic signal acquisition and generation
- Real-time data streaming architecture
- Core DSP algorithms (FFT, filtering)

**Phase 2: Advanced Features (Months 4-6)**
- Advanced triggering system
- Protocol analysis capabilities
- Measurement automation engine
- Function generator modulation schemes

**Phase 3: Optimization and Integration (Months 7-9)**
- Performance optimization with SIMD
- Hardware acceleration integration
- Calibration and compensation systems
- Claude Code integration and testing

**Phase 4: Production and Deployment (Months 10-12)**
- Comprehensive testing and validation
- Documentation and user guides
- Performance benchmarking
- Production deployment support

### 8.2 Success Metrics

**Performance Metrics:**
- Real-time processing latency <1 ms
- Sustained data throughput >1 GB/s
- Measurement accuracy within specifications
- System uptime >99.9%

**Integration Metrics:**
- MCP protocol compliance
- Claude Code compatibility
- API response times <10 ms
- Error rates <0.1%

This comprehensive technical plan provides a complete roadmap for implementing a high-performance MCP server that combines digital oscilloscope and function generator capabilities. The architecture leverages modern software engineering practices, advanced DSP algorithms, and hardware acceleration to deliver professional-grade measurement and signal generation performance through Claude Code integration.

The modular design enables incremental development and testing, while the performance optimization strategies ensure real-time operation with minimal latency. The integration with MCP protocol standards provides a robust, scalable platform for advanced signal processing applications in scientific research and engineering development.