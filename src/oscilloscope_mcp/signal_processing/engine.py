"""
Main signal processing engine for oscilloscope data.

This module provides the core signal processing functionality including
waveform processing, FFT analysis, and signal conditioning.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal
import structlog

from .fft import FFTAnalyzer
from .filters import DigitalFilter
from .windowing import WindowFunction


logger = structlog.get_logger(__name__)


class SignalProcessor:
    """Main signal processing engine."""
    
    def __init__(self):
        self.fft_analyzer = FFTAnalyzer()
        self.digital_filter = DigitalFilter()
        self.window_function = WindowFunction()
        self.is_initialized = False
        
        # Processing parameters
        self.default_sample_rate = 100e6  # 100 MS/s
        self.default_resolution = 1024
        self.noise_floor = -80  # dBFS
        
    async def initialize(self) -> None:
        """Initialize the signal processor."""
        logger.info("Initializing signal processor")
        
        try:
            # Initialize sub-components
            await self.fft_analyzer.initialize()
            await self.digital_filter.initialize()
            await self.window_function.initialize()
            
            self.is_initialized = True
            logger.info("Signal processor initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize signal processor", error=str(e))
            raise
    
    async def process_waveform(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw waveform data from hardware."""
        logger.info("Processing waveform data")
        
        if not self.is_initialized:
            raise RuntimeError("Signal processor not initialized")
        
        try:
            # Extract metadata
            metadata = raw_data.get("metadata", {})
            sample_rate = metadata.get("sample_rate", self.default_sample_rate)
            
            # Extract waveform data
            waveform_data = raw_data.get("waveform_data", {})
            time_axis = np.array(waveform_data.get("time", []))
            channels = waveform_data.get("channels", {})
            
            # Process each channel
            processed_channels = {}
            for ch_num, ch_data in channels.items():
                voltage_data = np.array(ch_data.get("voltage", []))
                
                if len(voltage_data) == 0:
                    continue
                
                # Apply signal conditioning
                conditioned_data = await self._condition_signal(
                    voltage_data, 
                    sample_rate,
                    ch_data
                )
                
                processed_channels[ch_num] = {
                    "voltage": conditioned_data.tolist(),
                    "voltage_range": ch_data.get("voltage_range"),
                    "coupling": ch_data.get("coupling"),
                    "impedance": ch_data.get("impedance"),
                    "processing_info": {
                        "conditioning_applied": True,
                        "sample_rate": sample_rate,
                        "samples": len(conditioned_data)
                    }
                }
            
            # Create processed waveform structure
            processed_data = {
                "time": time_axis.tolist(),
                "channels": processed_channels,
                "sample_rate": sample_rate,
                "timestamp": time.time(),
                "processing_metadata": {
                    "processor_version": "1.0.0",
                    "processing_time": time.time(),
                    "original_samples": len(time_axis)
                }
            }
            
            logger.info("Waveform processing completed", 
                       channels_processed=len(processed_channels),
                       samples=len(time_axis))
            
            return processed_data
            
        except Exception as e:
            logger.error("Failed to process waveform", error=str(e))
            raise
    
    async def _condition_signal(
        self, 
        signal_data: np.ndarray, 
        sample_rate: float,
        channel_config: Dict[str, Any]
    ) -> np.ndarray:
        """Apply signal conditioning to raw data."""
        
        conditioned = signal_data.copy()
        
        # Remove DC offset if AC coupling
        if channel_config.get("coupling") == "AC":
            conditioned = conditioned - np.mean(conditioned)
        
        # Apply anti-aliasing filter if needed
        nyquist = sample_rate / 2
        if sample_rate > 1e6:  # Apply for high sample rates
            # Design low-pass filter at 80% of Nyquist
            cutoff = 0.8 * nyquist
            conditioned = await self.digital_filter.apply_lowpass_filter(
                conditioned, sample_rate, cutoff
            )
        
        return conditioned
    
    async def analyze_spectrum(
        self,
        waveform_data: Dict[str, Any],
        window: str = "hamming",
        resolution: int = 1024,
        overlap: float = 0.5
    ) -> Dict[str, Any]:
        """Perform FFT analysis on waveform data."""
        logger.info("Analyzing spectrum", window=window, resolution=resolution)
        
        if not self.is_initialized:
            raise RuntimeError("Signal processor not initialized")
        
        try:
            # Get the first available channel for analysis
            channels = waveform_data.get("channels", {})
            if not channels:
                raise ValueError("No channel data available for spectrum analysis")
            
            # Use the first channel
            channel_num = list(channels.keys())[0]
            channel_data = channels[channel_num]
            voltage_data = np.array(channel_data.get("voltage", []))
            
            if len(voltage_data) == 0:
                raise ValueError("No voltage data available for analysis")
            
            sample_rate = waveform_data.get("sample_rate", self.default_sample_rate)
            
            # Perform FFT analysis
            spectrum_result = await self.fft_analyzer.compute_fft(
                signal_data=voltage_data,
                sample_rate=sample_rate,
                window=window,
                nfft=resolution,
                overlap=overlap
            )
            
            # Add metadata
            spectrum_result["analysis_metadata"] = {
                "channel": channel_num,
                "window": window,
                "resolution": resolution,
                "overlap": overlap,
                "sample_rate": sample_rate,
                "samples_analyzed": len(voltage_data),
                "timestamp": time.time()
            }
            
            logger.info("Spectrum analysis completed", 
                       channel=channel_num,
                       frequency_bins=len(spectrum_result.get("frequency", [])))
            
            return spectrum_result
            
        except Exception as e:
            logger.error("Failed to analyze spectrum", error=str(e))
            raise
    
    async def apply_digital_filter(
        self,
        waveform_data: Dict[str, Any],
        filter_type: str,
        cutoff_freq: Union[float, List[float]],
        order: int = 4
    ) -> Dict[str, Any]:
        """Apply digital filter to waveform data."""
        logger.info("Applying digital filter", 
                   filter_type=filter_type, 
                   cutoff_freq=cutoff_freq, 
                   order=order)
        
        if not self.is_initialized:
            raise RuntimeError("Signal processor not initialized")
        
        try:
            sample_rate = waveform_data.get("sample_rate", self.default_sample_rate)
            channels = waveform_data.get("channels", {})
            
            # Process each channel
            filtered_channels = {}
            for ch_num, ch_data in channels.items():
                voltage_data = np.array(ch_data.get("voltage", []))
                
                if len(voltage_data) == 0:
                    continue
                
                # Apply appropriate filter
                if filter_type == "lowpass":
                    filtered_data = await self.digital_filter.apply_lowpass_filter(
                        voltage_data, sample_rate, cutoff_freq, order
                    )
                elif filter_type == "highpass":
                    filtered_data = await self.digital_filter.apply_highpass_filter(
                        voltage_data, sample_rate, cutoff_freq, order
                    )
                elif filter_type == "bandpass":
                    if not isinstance(cutoff_freq, list) or len(cutoff_freq) != 2:
                        raise ValueError("Bandpass filter requires two cutoff frequencies")
                    filtered_data = await self.digital_filter.apply_bandpass_filter(
                        voltage_data, sample_rate, cutoff_freq, order
                    )
                elif filter_type == "bandstop":
                    if not isinstance(cutoff_freq, list) or len(cutoff_freq) != 2:
                        raise ValueError("Bandstop filter requires two cutoff frequencies")
                    filtered_data = await self.digital_filter.apply_bandstop_filter(
                        voltage_data, sample_rate, cutoff_freq, order
                    )
                else:
                    raise ValueError(f"Unsupported filter type: {filter_type}")
                
                # Update channel data
                filtered_channels[ch_num] = ch_data.copy()
                filtered_channels[ch_num]["voltage"] = filtered_data.tolist()
                filtered_channels[ch_num]["filter_applied"] = {
                    "type": filter_type,
                    "cutoff_freq": cutoff_freq,
                    "order": order
                }
            
            # Create filtered waveform data
            filtered_waveform = waveform_data.copy()
            filtered_waveform["channels"] = filtered_channels
            filtered_waveform["processing_metadata"]["filter_applied"] = {
                "type": filter_type,
                "cutoff_freq": cutoff_freq,
                "order": order,
                "timestamp": time.time()
            }
            
            logger.info("Digital filter applied successfully", 
                       channels_processed=len(filtered_channels))
            
            return filtered_waveform
            
        except Exception as e:
            logger.error("Failed to apply digital filter", error=str(e))
            raise
    
    async def calculate_cross_correlation(
        self,
        waveform_data1: Dict[str, Any],
        waveform_data2: Dict[str, Any],
        channel1: int,
        channel2: int
    ) -> Dict[str, Any]:
        """Calculate cross-correlation between two channels."""
        logger.info("Calculating cross-correlation", channel1=channel1, channel2=channel2)
        
        try:
            # Extract voltage data
            ch1_data = waveform_data1["channels"][channel1]["voltage"]
            ch2_data = waveform_data2["channels"][channel2]["voltage"]
            
            signal1 = np.array(ch1_data)
            signal2 = np.array(ch2_data)
            
            # Ensure signals have the same length
            min_length = min(len(signal1), len(signal2))
            signal1 = signal1[:min_length]
            signal2 = signal2[:min_length]
            
            # Calculate cross-correlation
            correlation = np.correlate(signal1, signal2, mode='full')
            
            # Calculate lag axis
            sample_rate = waveform_data1.get("sample_rate", self.default_sample_rate)
            lags = np.arange(-min_length + 1, min_length) / sample_rate
            
            # Find peak correlation and delay
            max_corr_idx = np.argmax(np.abs(correlation))
            max_correlation = correlation[max_corr_idx]
            delay = lags[max_corr_idx]
            
            result = {
                "correlation": correlation.tolist(),
                "lags": lags.tolist(),
                "max_correlation": float(max_correlation),
                "delay_seconds": float(delay),
                "delay_samples": int(lags[max_corr_idx] * sample_rate),
                "analysis_metadata": {
                    "channel1": channel1,
                    "channel2": channel2,
                    "samples_analyzed": min_length,
                    "sample_rate": sample_rate,
                    "timestamp": time.time()
                }
            }
            
            logger.info("Cross-correlation completed", 
                       max_correlation=max_correlation,
                       delay_seconds=delay)
            
            return result
            
        except Exception as e:
            logger.error("Failed to calculate cross-correlation", error=str(e))
            raise
    
    async def calculate_transfer_function(
        self,
        input_waveform: Dict[str, Any],
        output_waveform: Dict[str, Any],
        input_channel: int,
        output_channel: int
    ) -> Dict[str, Any]:
        """Calculate system transfer function H(f) = Output/Input."""
        logger.info("Calculating transfer function", 
                   input_channel=input_channel, 
                   output_channel=output_channel)
        
        try:
            # Extract voltage data
            input_data = np.array(input_waveform["channels"][input_channel]["voltage"])
            output_data = np.array(output_waveform["channels"][output_channel]["voltage"])
            
            # Ensure signals have the same length
            min_length = min(len(input_data), len(output_data))
            input_data = input_data[:min_length]
            output_data = output_data[:min_length]
            
            sample_rate = input_waveform.get("sample_rate", self.default_sample_rate)
            
            # Calculate FFTs
            input_fft = await self.fft_analyzer.compute_fft(
                input_data, sample_rate, window="hamming"
            )
            output_fft = await self.fft_analyzer.compute_fft(
                output_data, sample_rate, window="hamming"
            )
            
            # Calculate transfer function
            input_complex = np.array(input_fft["complex"])
            output_complex = np.array(output_fft["complex"])
            
            # Avoid division by zero
            transfer_function = np.divide(
                output_complex,
                input_complex,
                out=np.zeros_like(output_complex),
                where=np.abs(input_complex) > 1e-10
            )
            
            # Calculate magnitude and phase
            magnitude = np.abs(transfer_function)
            phase = np.angle(transfer_function, deg=True)
            magnitude_db = 20 * np.log10(magnitude + 1e-10)
            
            result = {
                "frequency": input_fft["frequency"],
                "magnitude": magnitude.tolist(),
                "magnitude_db": magnitude_db.tolist(),
                "phase_deg": phase.tolist(),
                "complex": transfer_function.tolist(),
                "analysis_metadata": {
                    "input_channel": input_channel,
                    "output_channel": output_channel,
                    "samples_analyzed": min_length,
                    "sample_rate": sample_rate,
                    "timestamp": time.time()
                }
            }
            
            logger.info("Transfer function calculated successfully")
            
            return result
            
        except Exception as e:
            logger.error("Failed to calculate transfer function", error=str(e))
            raise
    
    async def detect_signal_anomalies(
        self,
        waveform_data: Dict[str, Any],
        channel: int,
        threshold_sigma: float = 3.0
    ) -> Dict[str, Any]:
        """Detect signal anomalies using statistical analysis."""
        logger.info("Detecting signal anomalies", channel=channel, threshold_sigma=threshold_sigma)
        
        try:
            voltage_data = np.array(waveform_data["channels"][channel]["voltage"])
            sample_rate = waveform_data.get("sample_rate", self.default_sample_rate)
            time_axis = np.array(waveform_data.get("time", []))
            
            # Calculate statistics
            mean_val = np.mean(voltage_data)
            std_val = np.std(voltage_data)
            threshold = threshold_sigma * std_val
            
            # Find anomalies
            anomaly_indices = np.where(np.abs(voltage_data - mean_val) > threshold)[0]
            
            # Group consecutive anomalies
            anomaly_groups = []
            if len(anomaly_indices) > 0:
                start_idx = anomaly_indices[0]
                end_idx = start_idx
                
                for i in range(1, len(anomaly_indices)):
                    if anomaly_indices[i] == anomaly_indices[i-1] + 1:
                        end_idx = anomaly_indices[i]
                    else:
                        # End of current group
                        anomaly_groups.append({
                            "start_index": int(start_idx),
                            "end_index": int(end_idx),
                            "start_time": float(time_axis[start_idx]) if len(time_axis) > start_idx else None,
                            "end_time": float(time_axis[end_idx]) if len(time_axis) > end_idx else None,
                            "duration": float((end_idx - start_idx + 1) / sample_rate),
                            "max_deviation": float(np.max(np.abs(voltage_data[start_idx:end_idx+1] - mean_val)))
                        })
                        start_idx = anomaly_indices[i]
                        end_idx = start_idx
                
                # Add the last group
                anomaly_groups.append({
                    "start_index": int(start_idx),
                    "end_index": int(end_idx),
                    "start_time": float(time_axis[start_idx]) if len(time_axis) > start_idx else None,
                    "end_time": float(time_axis[end_idx]) if len(time_axis) > end_idx else None,
                    "duration": float((end_idx - start_idx + 1) / sample_rate),
                    "max_deviation": float(np.max(np.abs(voltage_data[start_idx:end_idx+1] - mean_val)))
                })
            
            result = {
                "anomalies_detected": len(anomaly_groups),
                "anomaly_groups": anomaly_groups,
                "statistics": {
                    "mean": float(mean_val),
                    "std_dev": float(std_val),
                    "threshold": float(threshold),
                    "threshold_sigma": threshold_sigma
                },
                "analysis_metadata": {
                    "channel": channel,
                    "samples_analyzed": len(voltage_data),
                    "sample_rate": sample_rate,
                    "timestamp": time.time()
                }
            }
            
            logger.info("Anomaly detection completed", 
                       anomalies_detected=len(anomaly_groups))
            
            return result
            
        except Exception as e:
            logger.error("Failed to detect signal anomalies", error=str(e))
            raise
    
    async def cleanup(self) -> None:
        """Cleanup signal processor resources."""
        logger.info("Cleaning up signal processor")
        
        try:
            # Cleanup sub-components
            if hasattr(self.fft_analyzer, 'cleanup'):
                await self.fft_analyzer.cleanup()
            if hasattr(self.digital_filter, 'cleanup'):
                await self.digital_filter.cleanup()
            if hasattr(self.window_function, 'cleanup'):
                await self.window_function.cleanup()
            
            self.is_initialized = False
            logger.info("Signal processor cleaned up successfully")
            
        except Exception as e:
            logger.error("Error during signal processor cleanup", error=str(e))
            raise