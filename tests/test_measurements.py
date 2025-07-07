"""
Test measurement analysis functionality.

Tests for automated measurements, parameter extraction, and measurement statistics.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from oscilloscope_mcp.measurements.analyzer import MeasurementAnalyzer


class TestMeasurementAnalyzer:
    """Test measurement analyzer functionality."""

    def test_analyzer_creation(self):
        """Test measurement analyzer instantiation."""
        analyzer = MeasurementAnalyzer()
        assert analyzer is not None

    def test_rms_measurement(self):
        """Test RMS voltage measurement."""
        analyzer = MeasurementAnalyzer()
        
        # Test with known RMS values
        # DC signal: RMS = DC value
        dc_signal = np.ones(1000) * 2.5
        rms_dc = analyzer.measure_rms(dc_signal.tolist())
        assert rms_dc == pytest.approx(2.5, abs=1e-6)
        
        # Sine wave: RMS = amplitude / sqrt(2)
        sample_rate = 1000
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        amplitude = 3.0
        sine_signal = amplitude * np.sin(2 * np.pi * 50 * t)
        rms_sine = analyzer.measure_rms(sine_signal.tolist())
        expected_rms = amplitude / np.sqrt(2)
        assert rms_sine == pytest.approx(expected_rms, abs=0.01)

    def test_peak_to_peak_measurement(self):
        """Test peak-to-peak voltage measurement."""
        analyzer = MeasurementAnalyzer()
        
        # Square wave with known amplitude
        amplitude = 4.0
        offset = 1.0
        signal = np.concatenate([
            np.ones(500) * (amplitude + offset),
            np.ones(500) * (-amplitude + offset)
        ])
        
        pk_pk = analyzer.measure_peak_to_peak(signal.tolist())
        assert pk_pk == pytest.approx(2 * amplitude, abs=1e-6)

    def test_frequency_measurement(self):
        """Test frequency measurement."""
        analyzer = MeasurementAnalyzer()
        
        # Generate sine wave with known frequency
        frequency = 125.5  # Non-integer frequency
        sample_rate = 2000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t)
        
        measured_freq = analyzer.measure_frequency(signal.tolist(), sample_rate)
        assert measured_freq == pytest.approx(frequency, abs=1.0)

    def test_rise_time_measurement(self):
        """Test rise time measurement."""
        analyzer = MeasurementAnalyzer()
        
        # Create step response with known rise time
        sample_rate = 10000  # High sample rate for accuracy
        total_samples = 1000
        
        # Rise from 10% to 90% over 50 samples
        rise_samples = 50
        rise_time_expected = rise_samples / sample_rate
        
        signal = np.concatenate([
            np.zeros(100),  # Initial low
            np.linspace(0, 1, rise_samples),  # Rising edge
            np.ones(total_samples - 100 - rise_samples)  # Final high
        ])
        
        rise_time = analyzer.measure_rise_time(signal.tolist(), sample_rate)
        assert rise_time == pytest.approx(rise_time_expected, abs=rise_time_expected * 0.1)

    def test_fall_time_measurement(self):
        """Test fall time measurement."""
        analyzer = MeasurementAnalyzer()
        
        # Create falling edge
        sample_rate = 5000
        fall_samples = 25
        fall_time_expected = fall_samples / sample_rate
        
        signal = np.concatenate([
            np.ones(100),  # Initial high
            np.linspace(1, 0, fall_samples),  # Falling edge
            np.zeros(200)  # Final low
        ])
        
        fall_time = analyzer.measure_fall_time(signal.tolist(), sample_rate)
        assert fall_time == pytest.approx(fall_time_expected, abs=fall_time_expected * 0.15)

    def test_pulse_width_measurement(self):
        """Test pulse width measurement."""
        analyzer = MeasurementAnalyzer()
        
        # Create pulse with known width
        sample_rate = 1000
        pulse_width_samples = 100
        pulse_width_expected = pulse_width_samples / sample_rate
        
        signal = np.concatenate([
            np.zeros(50),  # Low
            np.ones(pulse_width_samples),  # High pulse
            np.zeros(50)   # Low
        ])
        
        pulse_width = analyzer.measure_pulse_width(signal.tolist(), sample_rate)
        assert pulse_width == pytest.approx(pulse_width_expected, abs=pulse_width_expected * 0.1)

    def test_duty_cycle_measurement(self):
        """Test duty cycle measurement."""
        analyzer = MeasurementAnalyzer()
        
        # Create square wave with 30% duty cycle
        period_samples = 100
        high_samples = 30
        duty_cycle_expected = high_samples / period_samples
        
        # Create multiple periods
        single_period = np.concatenate([
            np.ones(high_samples),
            np.zeros(period_samples - high_samples)
        ])
        signal = np.tile(single_period, 10)  # 10 periods
        
        sample_rate = 1000
        duty_cycle = analyzer.measure_duty_cycle(signal.tolist(), sample_rate)
        assert duty_cycle == pytest.approx(duty_cycle_expected, abs=0.05)

    def test_overshoot_measurement(self):
        """Test overshoot measurement."""
        analyzer = MeasurementAnalyzer()
        
        # Create step response with overshoot
        steady_state = 1.0
        overshoot_percent = 20.0  # 20% overshoot
        peak_value = steady_state * (1 + overshoot_percent / 100)
        
        signal = np.concatenate([
            np.zeros(50),  # Initial low
            [peak_value] * 10,  # Overshoot peak
            [steady_state] * 100  # Steady state
        ])
        
        overshoot = analyzer.measure_overshoot(signal.tolist())
        assert overshoot == pytest.approx(overshoot_percent, abs=2.0)

    def test_settling_time_measurement(self):
        """Test settling time measurement."""
        analyzer = MeasurementAnalyzer()
        
        # Create step response that settles within 2% after 80 samples
        sample_rate = 1000
        settling_samples = 80
        settling_time_expected = settling_samples / sample_rate
        
        final_value = 1.0
        tolerance = 0.02  # 2% tolerance
        
        # Exponential decay to final value
        t = np.arange(200)
        response = final_value * (1 - np.exp(-t / 20))  # Time constant = 20 samples
        
        # Prepend initial zeros
        signal = np.concatenate([np.zeros(20), response])
        
        settling_time = analyzer.measure_settling_time(
            signal.tolist(), 
            sample_rate, 
            tolerance_percent=2.0
        )
        
        # Should be close to expected settling time
        assert settling_time > 0
        assert settling_time < 0.5  # Should settle within 500ms


class TestMeasurementStatistics:
    """Test measurement statistics and analysis."""

    def test_measurement_statistics(self):
        """Test measurement statistics calculation."""
        analyzer = MeasurementAnalyzer()
        
        # Generate multiple measurements
        measurements = []
        for i in range(100):
            # Add some variation to the measurements
            base_value = 2.5
            variation = 0.1 * np.sin(i * 0.1) + 0.05 * np.random.randn()
            measurements.append(base_value + variation)
        
        stats = analyzer.calculate_statistics(measurements)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        
        assert stats["mean"] == pytest.approx(2.5, abs=0.2)
        assert stats["count"] == 100
        assert stats["std"] > 0
        assert stats["min"] < stats["max"]

    def test_measurement_trends(self):
        """Test measurement trend analysis."""
        analyzer = MeasurementAnalyzer()
        
        # Create trending data
        time_points = np.linspace(0, 10, 100)
        trend_slope = 0.1
        measurements = 2.0 + trend_slope * time_points + 0.05 * np.random.randn(100)
        
        trend_analysis = analyzer.analyze_trend(measurements.tolist(), time_points.tolist())
        
        assert "slope" in trend_analysis
        assert "correlation" in trend_analysis
        assert "trend_significance" in trend_analysis
        
        assert trend_analysis["slope"] == pytest.approx(trend_slope, abs=0.05)
        assert abs(trend_analysis["correlation"]) > 0.8  # Strong correlation

    def test_measurement_outlier_detection(self):
        """Test outlier detection in measurements."""
        analyzer = MeasurementAnalyzer()
        
        # Create data with outliers
        normal_data = np.random.normal(5.0, 0.5, 95)
        outliers = np.array([10.0, -2.0, 15.0, 0.5, 12.0])
        all_data = np.concatenate([normal_data, outliers])
        
        outlier_analysis = analyzer.detect_outliers(all_data.tolist())
        
        assert "outlier_indices" in outlier_analysis
        assert "outlier_values" in outlier_analysis
        assert "outlier_count" in outlier_analysis
        
        assert len(outlier_analysis["outlier_indices"]) > 0
        assert outlier_analysis["outlier_count"] > 0


class TestParameterExtraction:
    """Test parameter extraction from waveforms."""

    def test_amplitude_parameters(self):
        """Test amplitude-related parameter extraction."""
        analyzer = MeasurementAnalyzer()
        
        # Create signal with known amplitude characteristics
        offset = 1.5
        amplitude = 2.0
        sample_rate = 1000
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        signal = offset + amplitude * np.sin(2 * np.pi * 50 * t)
        
        params = analyzer.extract_amplitude_parameters(signal.tolist())
        
        assert "dc_component" in params
        assert "ac_component" in params
        assert "peak_positive" in params
        assert "peak_negative" in params
        
        assert params["dc_component"] == pytest.approx(offset, abs=0.1)
        assert params["peak_positive"] == pytest.approx(offset + amplitude, abs=0.1)
        assert params["peak_negative"] == pytest.approx(offset - amplitude, abs=0.1)

    def test_timing_parameters(self):
        """Test timing-related parameter extraction."""
        analyzer = MeasurementAnalyzer()
        
        # Create periodic signal
        frequency = 100  # Hz
        sample_rate = 10000
        periods = 5
        total_samples = int(periods * sample_rate / frequency)
        
        t = np.linspace(0, periods / frequency, total_samples, endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t)
        
        timing_params = analyzer.extract_timing_parameters(signal.tolist(), sample_rate)
        
        assert "frequency" in timing_params
        assert "period" in timing_params
        assert "zero_crossings" in timing_params
        
        assert timing_params["frequency"] == pytest.approx(frequency, abs=5.0)
        assert timing_params["period"] == pytest.approx(1.0 / frequency, abs=0.001)

    def test_signal_quality_parameters(self):
        """Test signal quality parameter extraction."""
        analyzer = MeasurementAnalyzer()
        
        # Create signal with noise
        sample_rate = 2000
        t = np.linspace(0, 0.5, sample_rate, endpoint=False)
        
        clean_signal = np.sin(2 * np.pi * 200 * t)
        noise = 0.1 * np.random.randn(len(t))
        noisy_signal = clean_signal + noise
        
        quality_params = analyzer.extract_quality_parameters(noisy_signal.tolist(), sample_rate)
        
        assert "snr_estimate" in quality_params
        assert "thd_estimate" in quality_params
        assert "noise_floor" in quality_params
        
        assert quality_params["snr_estimate"] > 0
        assert quality_params["noise_floor"] > 0


class TestMeasurementIntegration:
    """Integration tests for measurement functionality."""

    async def test_comprehensive_waveform_analysis(self):
        """Test comprehensive analysis of a complex waveform."""
        analyzer = MeasurementAnalyzer()
        
        # Create complex test waveform
        sample_rate = 5000
        duration = 0.2
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Fundamental + harmonics + noise
        fundamental = 2.0 * np.sin(2 * np.pi * 250 * t)
        harmonic2 = 0.5 * np.sin(2 * np.pi * 500 * t + np.pi/3)
        harmonic3 = 0.2 * np.sin(2 * np.pi * 750 * t - np.pi/4)
        noise = 0.1 * np.random.randn(len(t))
        dc_offset = 0.5
        
        signal = dc_offset + fundamental + harmonic2 + harmonic3 + noise
        
        # Perform comprehensive analysis
        measurements = analyzer.comprehensive_analysis(signal.tolist(), sample_rate)
        
        # Verify all measurement categories are present
        assert "amplitude_measurements" in measurements
        assert "timing_measurements" in measurements
        assert "frequency_measurements" in measurements
        assert "quality_measurements" in measurements
        
        # Verify specific measurements
        amp_measurements = measurements["amplitude_measurements"]
        assert "rms" in amp_measurements
        assert "peak_to_peak" in amp_measurements
        assert "dc_component" in amp_measurements
        
        timing_measurements = measurements["timing_measurements"]
        assert "frequency" in timing_measurements
        
        # DC component should be close to offset
        assert amp_measurements["dc_component"] == pytest.approx(dc_offset, abs=0.1)
        
        # Frequency should be close to fundamental
        assert timing_measurements["frequency"] == pytest.approx(250, abs=10)

    def test_multi_channel_measurement_correlation(self):
        """Test measurement correlation between multiple channels."""
        analyzer = MeasurementAnalyzer()
        
        # Create two correlated channels
        sample_rate = 1000
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        
        # Channel 1: original signal
        ch1 = np.sin(2 * np.pi * 100 * t)
        
        # Channel 2: phase-shifted and scaled version
        ch2 = 1.5 * np.sin(2 * np.pi * 100 * t + np.pi/4)
        
        channels_data = {
            1: ch1.tolist(),
            2: ch2.tolist()
        }
        
        correlation_analysis = analyzer.analyze_channel_correlation(channels_data, sample_rate)
        
        assert "cross_correlation" in correlation_analysis
        assert "phase_difference" in correlation_analysis
        assert "amplitude_ratio" in correlation_analysis
        
        # Should detect strong correlation
        assert abs(correlation_analysis["cross_correlation"]) > 0.8
        
        # Phase difference should be close to pi/4
        assert correlation_analysis["phase_difference"] == pytest.approx(np.pi/4, abs=0.2)
        
        # Amplitude ratio should be close to 1.5
        assert correlation_analysis["amplitude_ratio"] == pytest.approx(1.5, abs=0.2)

    def test_measurement_repeatability(self):
        """Test measurement repeatability and consistency."""
        analyzer = MeasurementAnalyzer()
        
        # Generate identical signals multiple times
        sample_rate = 2000
        t = np.linspace(0, 0.1, 200, endpoint=False)
        reference_signal = 3.0 * np.sin(2 * np.pi * 500 * t) + 1.0
        
        measurements = []
        for i in range(10):
            # Add tiny amount of noise to simulate real measurement variation
            noisy_signal = reference_signal + 0.001 * np.random.randn(len(reference_signal))
            
            result = analyzer.comprehensive_analysis(noisy_signal.tolist(), sample_rate)
            measurements.append(result)
        
        # Extract RMS values for consistency check
        rms_values = [m["amplitude_measurements"]["rms"] for m in measurements]
        
        # All measurements should be very close
        rms_std = np.std(rms_values)
        assert rms_std < 0.01  # Very low variation expected
        
        # Mean should be close to theoretical RMS
        theoretical_rms = np.sqrt((3.0/np.sqrt(2))**2 + 1.0**2)  # RMS of sine + DC
        measured_mean = np.mean(rms_values)
        assert measured_mean == pytest.approx(theoretical_rms, abs=0.05)