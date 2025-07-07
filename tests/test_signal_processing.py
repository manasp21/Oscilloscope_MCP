"""
Test signal processing functionality.

Tests for FFT analysis, filtering, windowing, and signal processing engine.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from oscilloscope_mcp.signal_processing.engine import SignalProcessor
from oscilloscope_mcp.signal_processing.fft import FFTAnalyzer
from oscilloscope_mcp.signal_processing.filters import DigitalFilter
from oscilloscope_mcp.signal_processing.windowing import WindowFunction


class TestSignalProcessor:
    """Test signal processing engine."""

    def test_signal_processor_creation(self):
        """Test signal processor instantiation."""
        processor = SignalProcessor()
        assert processor is not None

    async def test_basic_signal_analysis(self):
        """Test basic signal analysis functionality."""
        processor = SignalProcessor()
        
        # Create test signal: 1kHz sine wave
        sample_rate = 10000  # 10 kHz
        duration = 0.1  # 100ms
        frequency = 1000  # 1 kHz
        
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t)
        
        # Test signal analysis
        result = await processor.analyze_signal(signal.tolist(), sample_rate)
        
        assert "rms" in result
        assert "peak_to_peak" in result
        assert "frequency_estimate" in result
        assert result["rms"] > 0
        assert result["peak_to_peak"] > 0

    async def test_multi_channel_processing(self):
        """Test multi-channel signal processing."""
        processor = SignalProcessor()
        
        # Create two-channel test data
        sample_rate = 8000
        duration = 0.05
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Channel 1: 500 Hz sine wave
        ch1 = np.sin(2 * np.pi * 500 * t)
        # Channel 2: 1500 Hz sine wave
        ch2 = np.sin(2 * np.pi * 1500 * t)
        
        channels_data = {
            1: ch1.tolist(),
            2: ch2.tolist()
        }
        
        result = await processor.process_channels(channels_data, sample_rate)
        
        assert len(result) == 2
        assert 1 in result
        assert 2 in result
        assert "measurements" in result[1]
        assert "measurements" in result[2]


class TestFFTAnalyzer:
    """Test FFT analysis functionality."""

    def test_fft_analyzer_creation(self):
        """Test FFT analyzer instantiation."""
        analyzer = FFTAnalyzer()
        assert analyzer is not None

    def test_power_spectrum_calculation(self):
        """Test power spectrum calculation."""
        analyzer = FFTAnalyzer()
        
        # Create test signal with known frequency content
        sample_rate = 1000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # 100 Hz + 300 Hz signal
        signal = (np.sin(2 * np.pi * 100 * t) + 
                 0.5 * np.sin(2 * np.pi * 300 * t))
        
        frequencies, power_spectrum = analyzer.power_spectrum(
            signal, sample_rate, window="hann"
        )
        
        assert len(frequencies) == len(power_spectrum)
        assert len(frequencies) > 0
        assert np.max(power_spectrum) > 0
        
        # Find peaks around expected frequencies
        peak_indices = np.where(power_spectrum > 0.1 * np.max(power_spectrum))[0]
        peak_frequencies = frequencies[peak_indices]
        
        # Should have peaks near 100 Hz and 300 Hz
        assert len(peak_frequencies) >= 2

    def test_spectral_measurements(self):
        """Test spectral measurement calculations."""
        analyzer = FFTAnalyzer()
        
        # White noise signal
        np.random.seed(42)  # For reproducible tests
        signal = np.random.normal(0, 1, 1000)
        sample_rate = 1000
        
        measurements = analyzer.spectral_measurements(signal, sample_rate)
        
        assert "dominant_frequency" in measurements
        assert "bandwidth" in measurements
        assert "total_power" in measurements
        assert "snr_estimate" in measurements
        
        assert measurements["total_power"] > 0
        assert measurements["bandwidth"] > 0

    def test_cross_correlation(self):
        """Test cross-correlation analysis."""
        analyzer = FFTAnalyzer()
        
        # Create two related signals
        sample_rate = 1000
        t = np.linspace(0, 0.1, 100, endpoint=False)
        
        signal1 = np.sin(2 * np.pi * 50 * t)
        signal2 = np.sin(2 * np.pi * 50 * t + np.pi/4)  # Phase shifted
        
        correlation = analyzer.cross_correlate(signal1, signal2)
        
        assert len(correlation) > 0
        assert np.max(np.abs(correlation)) > 0.5  # Should be well correlated


class TestDigitalFilter:
    """Test digital filtering functionality."""

    def test_filter_creation(self):
        """Test digital filter instantiation."""
        filter_obj = DigitalFilter()
        assert filter_obj is not None

    def test_lowpass_filter(self):
        """Test low-pass filtering."""
        filter_obj = DigitalFilter()
        
        # Create test signal: low freq + high freq
        sample_rate = 1000
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        
        # 10 Hz signal + 200 Hz noise
        low_freq = np.sin(2 * np.pi * 10 * t)
        high_freq = 0.3 * np.sin(2 * np.pi * 200 * t)
        signal = low_freq + high_freq
        
        # Apply low-pass filter at 50 Hz
        filtered = filter_obj.lowpass(signal, cutoff=50, sample_rate=sample_rate)
        
        assert len(filtered) == len(signal)
        
        # High frequency content should be reduced
        signal_fft = np.abs(np.fft.fft(signal))
        filtered_fft = np.abs(np.fft.fft(filtered))
        
        # Check that high frequency content is attenuated
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        high_freq_idx = np.where(np.abs(freqs) > 100)[0]
        
        if len(high_freq_idx) > 0:
            high_freq_attenuation = (
                np.mean(filtered_fft[high_freq_idx]) / 
                np.mean(signal_fft[high_freq_idx])
            )
            assert high_freq_attenuation < 0.8  # At least 20% attenuation

    def test_bandpass_filter(self):
        """Test band-pass filtering."""
        filter_obj = DigitalFilter()
        
        # Create multi-frequency signal
        sample_rate = 1000
        t = np.linspace(0, 0.5, 500, endpoint=False)
        
        signal = (np.sin(2 * np.pi * 10 * t) +    # Low freq
                 np.sin(2 * np.pi * 100 * t) +    # Target freq
                 np.sin(2 * np.pi * 400 * t))     # High freq
        
        # Band-pass filter: 50-150 Hz
        filtered = filter_obj.bandpass(
            signal, 
            low_cutoff=50, 
            high_cutoff=150, 
            sample_rate=sample_rate
        )
        
        assert len(filtered) == len(signal)
        
        # Target frequency (100 Hz) should be preserved
        # Other frequencies should be attenuated

    def test_notch_filter(self):
        """Test notch filtering."""
        filter_obj = DigitalFilter()
        
        # Create signal with 60 Hz interference
        sample_rate = 1000
        t = np.linspace(0, 0.2, 200, endpoint=False)
        
        desired_signal = np.sin(2 * np.pi * 50 * t)
        interference = 0.5 * np.sin(2 * np.pi * 60 * t)
        signal = desired_signal + interference
        
        # Apply 60 Hz notch filter
        filtered = filter_obj.notch(signal, notch_freq=60, sample_rate=sample_rate)
        
        assert len(filtered) == len(signal)
        
        # 60 Hz component should be reduced
        original_60hz = np.max(np.abs(np.fft.fft(signal * np.sin(2 * np.pi * 60 * t))))
        filtered_60hz = np.max(np.abs(np.fft.fft(filtered * np.sin(2 * np.pi * 60 * t))))
        
        assert filtered_60hz < original_60hz


class TestWindowFunction:
    """Test windowing functions."""

    def test_window_creation(self):
        """Test window function instantiation."""
        window = WindowFunction()
        assert window is not None

    def test_hann_window(self):
        """Test Hann window generation."""
        window = WindowFunction()
        
        length = 100
        hann_window = window.hann(length)
        
        assert len(hann_window) == length
        assert hann_window[0] == pytest.approx(0.0, abs=1e-10)
        assert hann_window[-1] == pytest.approx(0.0, abs=1e-10)
        assert np.max(hann_window) <= 1.0

    def test_blackman_window(self):
        """Test Blackman window generation."""
        window = WindowFunction()
        
        length = 256
        blackman_window = window.blackman(length)
        
        assert len(blackman_window) == length
        assert blackman_window[0] == pytest.approx(0.0, abs=1e-10)
        assert blackman_window[-1] == pytest.approx(0.0, abs=1e-10)
        assert np.max(blackman_window) <= 1.0

    def test_window_application(self):
        """Test applying window to signal."""
        window = WindowFunction()
        
        # Create test signal
        signal = np.ones(100)  # DC signal
        
        # Apply Hann window
        windowed = window.apply_window(signal, "hann")
        
        assert len(windowed) == len(signal)
        assert windowed[0] == pytest.approx(0.0, abs=1e-10)
        assert windowed[-1] == pytest.approx(0.0, abs=1e-10)
        assert np.max(windowed) < 1.0  # Should be attenuated

    def test_window_correction_factor(self):
        """Test window correction factor calculation."""
        window = WindowFunction()
        
        hann_factor = window.correction_factor("hann", 100)
        blackman_factor = window.correction_factor("blackman", 100)
        
        assert hann_factor > 1.0  # Compensation for window attenuation
        assert blackman_factor > hann_factor  # Blackman has more attenuation


class TestSignalProcessingIntegration:
    """Integration tests for signal processing components."""

    async def test_complete_spectral_analysis_workflow(self):
        """Test complete spectral analysis workflow."""
        processor = SignalProcessor()
        fft_analyzer = FFTAnalyzer()
        filter_obj = DigitalFilter()
        window = WindowFunction()
        
        # Create complex test signal
        sample_rate = 2000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Signal: 100 Hz sine + 500 Hz sine + noise
        signal = (np.sin(2 * np.pi * 100 * t) + 
                 0.5 * np.sin(2 * np.pi * 500 * t) +
                 0.1 * np.random.normal(0, 1, len(t)))
        
        # 1. Pre-filter the signal
        filtered_signal = filter_obj.lowpass(signal, cutoff=600, sample_rate=sample_rate)
        
        # 2. Apply windowing
        windowed_signal = window.apply_window(filtered_signal, "hann")
        
        # 3. Perform FFT analysis
        frequencies, power_spectrum = fft_analyzer.power_spectrum(
            windowed_signal, sample_rate, window="hann"
        )
        
        # 4. Get spectral measurements
        measurements = fft_analyzer.spectral_measurements(windowed_signal, sample_rate)
        
        # 5. Perform time-domain analysis
        time_analysis = await processor.analyze_signal(filtered_signal.tolist(), sample_rate)
        
        # Verify all components worked
        assert len(frequencies) > 0
        assert len(power_spectrum) > 0
        assert "dominant_frequency" in measurements
        assert "rms" in time_analysis
        assert time_analysis["rms"] > 0

    def test_filter_cascade(self):
        """Test cascading multiple filters."""
        filter_obj = DigitalFilter()
        
        # Create signal with multiple frequency components
        sample_rate = 1000
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        
        signal = (np.sin(2 * np.pi * 25 * t) +    # Keep this
                 np.sin(2 * np.pi * 60 * t) +     # Remove this (notch)
                 np.sin(2 * np.pi * 200 * t))     # Remove this (lowpass)
        
        # Apply cascade: lowpass then notch
        step1 = filter_obj.lowpass(signal, cutoff=100, sample_rate=sample_rate)
        step2 = filter_obj.notch(step1, notch_freq=60, sample_rate=sample_rate)
        
        assert len(step2) == len(signal)
        
        # Final signal should primarily contain 25 Hz component

    async def test_real_time_processing_simulation(self):
        """Test simulated real-time processing workflow."""
        processor = SignalProcessor()
        
        # Simulate incoming data chunks
        sample_rate = 1000
        chunk_size = 100
        total_chunks = 10
        
        results = []
        
        for i in range(total_chunks):
            # Generate chunk of data
            t_start = i * chunk_size / sample_rate
            t_end = (i + 1) * chunk_size / sample_rate
            t = np.linspace(t_start, t_end, chunk_size, endpoint=False)
            
            chunk = np.sin(2 * np.pi * 50 * t)  # 50 Hz signal
            
            # Process chunk
            result = await processor.analyze_signal(chunk.tolist(), sample_rate)
            results.append(result)
        
        # All chunks should be processed successfully
        assert len(results) == total_chunks
        for result in results:
            assert "rms" in result
            assert result["rms"] > 0