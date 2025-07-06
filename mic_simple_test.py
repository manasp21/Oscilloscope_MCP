#!/usr/bin/env python3
"""
Simplified Microphone Test - Minimal Dependencies

This version avoids the typer dependency conflict by implementing
just the core signal processing without the full MCP server stack.

Requirements (minimal):
- sounddevice: pip install sounddevice
- numpy: pip install numpy  
- matplotlib: pip install matplotlib
- scipy: pip install scipy

Usage:
    python mic_simple_test.py
"""

import sys
import time
import threading
import queue
import numpy as np
from typing import Dict, Any, List

try:
    import sounddevice as sd
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from scipy import signal
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nInstall required packages:")
    print("pip install sounddevice matplotlib numpy scipy")
    sys.exit(1)


class SimpleMicrophoneOscilloscope:
    """Simplified microphone oscilloscope with core signal processing."""
    
    def __init__(self):
        # Audio parameters
        self.sample_rate = 44100  # Standard audio sample rate
        self.channels = 1  # Mono audio
        self.buffer_size = 1024  # Audio buffer size
        self.dtype = np.float32
        
        # Data storage
        self.audio_queue = queue.Queue()
        self.latest_data = np.zeros(self.buffer_size)
        self.is_recording = False
        
        # Signal processing parameters
        self.window_type = "hamming"
        self.fft_size = 1024
        
        # Real-time plotting
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Microphone Oscilloscope - Core Signal Processing Test', fontsize=14, fontweight='bold')
        
        # Setup plots
        self.setup_plots()
        
    def setup_plots(self):
        """Setup matplotlib plots for real-time display."""
        
        # Time domain plot
        self.ax_time = self.axes[0, 0]
        self.ax_time.set_title('Time Domain Waveform')
        self.ax_time.set_xlabel('Time (s)')
        self.ax_time.set_ylabel('Amplitude')
        self.ax_time.grid(True, alpha=0.3)
        self.line_time, = self.ax_time.plot([], [], 'b-', linewidth=1)
        
        # Frequency domain plot
        self.ax_freq = self.axes[0, 1]
        self.ax_freq.set_title('Frequency Spectrum (FFT)')
        self.ax_freq.set_xlabel('Frequency (Hz)')
        self.ax_freq.set_ylabel('Magnitude (dB)')
        self.ax_freq.grid(True, alpha=0.3)
        self.line_freq, = self.ax_freq.plot([], [], 'r-', linewidth=1)
        
        # Measurements display
        self.ax_measurements = self.axes[1, 0]
        self.ax_measurements.set_title('Signal Measurements')
        self.ax_measurements.axis('off')
        self.measurements_text = self.ax_measurements.text(0.1, 0.5, '', 
                                                          transform=self.ax_measurements.transAxes,
                                                          fontsize=10, verticalalignment='center',
                                                          fontfamily='monospace')
        
        # Status display
        self.ax_status = self.axes[1, 1]
        self.ax_status.set_title('System Status')
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(0.1, 0.5, '', 
                                              transform=self.ax_status.transAxes,
                                              fontsize=10, verticalalignment='center',
                                              fontfamily='monospace')
        
        plt.tight_layout()
    
    def process_audio_signal(self, audio_data):
        """Core signal processing - simulates MCP oscilloscope processing."""
        try:
            # Basic signal conditioning (like MCP would do)
            # Remove DC component (AC coupling simulation)
            conditioned_signal = audio_data - np.mean(audio_data)
            
            # Apply window for FFT
            windowed_signal = conditioned_signal * np.hamming(len(conditioned_signal))
            
            # Compute FFT
            fft_result = np.fft.fft(windowed_signal, n=self.fft_size)
            freqs = np.fft.fftfreq(self.fft_size, 1/self.sample_rate)
            
            # Only keep positive frequencies
            positive_freqs = freqs[:self.fft_size//2]
            positive_fft = fft_result[:self.fft_size//2]
            
            # Calculate magnitude and phase
            magnitude = np.abs(positive_fft)
            magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Add small value to avoid log(0)
            phase_deg = np.angle(positive_fft, deg=True)
            
            # Calculate measurements (like MCP measurement analyzer would do)
            measurements = self.calculate_measurements(conditioned_signal)
            
            return {
                "success": True,
                "time_data": conditioned_signal,
                "frequency": positive_freqs,
                "magnitude_db": magnitude_db,
                "phase_deg": phase_deg,
                "measurements": measurements
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def calculate_measurements(self, signal_data):
        """Calculate oscilloscope-style measurements."""
        try:
            # RMS (Root Mean Square)
            rms = np.sqrt(np.mean(signal_data**2))
            
            # Peak-to-peak
            peak_to_peak = np.max(signal_data) - np.min(signal_data)
            
            # Maximum amplitude
            amplitude = np.max(np.abs(signal_data))
            
            # Frequency estimation using zero crossings
            zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
            if len(zero_crossings) > 2:
                # Calculate period from zero crossings
                periods = np.diff(zero_crossings) * 2  # *2 because we're counting half periods
                avg_period_samples = np.mean(periods)
                frequency = self.sample_rate / avg_period_samples
            else:
                frequency = 0.0
            
            # Signal-to-noise ratio estimate
            signal_power = np.var(signal_data)
            noise_estimate = np.var(signal_data - signal.medfilt(signal_data, kernel_size=5))
            snr_db = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
            
            return {
                "rms": float(rms),
                "peak_to_peak": float(peak_to_peak),
                "amplitude": float(amplitude),
                "frequency": float(frequency),
                "snr_db": float(snr_db)
            }
            
        except Exception as e:
            return {
                "rms": 0.0,
                "peak_to_peak": 0.0,
                "amplitude": 0.0,
                "frequency": 0.0,
                "snr_db": 0.0,
                "error": str(e)
            }
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input."""
        if status:
            print(f"Audio status: {status}")
        
        # Convert to mono if stereo
        if len(indata.shape) > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()
        
        # Add to queue for processing
        try:
            self.audio_queue.put_nowait(audio_data.copy())
        except queue.Full:
            # Skip if queue is full
            pass
    
    def get_audio_devices(self):
        """List available audio input devices."""
        print("\n🎤 Available Audio Input Devices:")
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (max inputs: {device['max_input_channels']})")
                input_devices.append(i)
        
        return input_devices
    
    def select_audio_device(self):
        """Let user select audio input device."""
        input_devices = self.get_audio_devices()
        
        if not input_devices:
            print("❌ No audio input devices found!")
            return None
        
        try:
            choice = input(f"\nSelect device (0-{len(sd.query_devices())-1}, or press Enter for default): ")
            if choice.strip() == "":
                return None  # Use default
            
            device_id = int(choice)
            if device_id in input_devices:
                return device_id
            else:
                print("❌ Invalid selection, using default device")
                return None
                
        except ValueError:
            print("❌ Invalid input, using default device")
            return None
    
    def update_plots(self, frame):
        """Update plots with latest data."""
        try:
            # Get latest audio data
            if not self.audio_queue.empty():
                self.latest_data = self.audio_queue.get_nowait()
            
            if len(self.latest_data) == 0:
                return self.line_time, self.line_freq
            
            # Process audio through core signal processing
            result = self.process_audio_signal(self.latest_data)
            
            if result["success"]:
                # Update time domain plot
                time_axis = np.linspace(0, len(self.latest_data) / self.sample_rate, len(self.latest_data))
                processed_data = result["time_data"]
                
                self.line_time.set_data(time_axis, processed_data)
                self.ax_time.set_xlim(0, max(time_axis))
                
                # Auto-scale Y axis based on signal amplitude
                y_max = max(0.1, np.max(np.abs(processed_data)) * 1.2)
                self.ax_time.set_ylim(-y_max, y_max)
                
                # Update frequency domain plot
                freq_data = result["frequency"]
                mag_data = result["magnitude_db"]
                
                # Only show up to 8 kHz for audio
                max_freq_idx = np.searchsorted(freq_data, 8000)
                freq_data = freq_data[:max_freq_idx]
                mag_data = mag_data[:max_freq_idx]
                
                self.line_freq.set_data(freq_data, mag_data)
                self.ax_freq.set_xlim(0, 8000)
                self.ax_freq.set_ylim(-80, 20)
                
                # Update measurements
                measurements = result["measurements"]
                measurements_str = "Signal Processing Results:\n\n"
                measurements_str += f"RMS: {measurements.get('rms', 0):.4f}\n"
                measurements_str += f"Peak-to-Peak: {measurements.get('peak_to_peak', 0):.4f}\n"
                measurements_str += f"Frequency: {measurements.get('frequency', 0):.1f} Hz\n"
                measurements_str += f"Amplitude: {measurements.get('amplitude', 0):.4f}\n"
                measurements_str += f"SNR: {measurements.get('snr_db', 0):.1f} dB\n"
                
                if "error" in measurements:
                    measurements_str += f"\n⚠️ Warning: {measurements['error']}"
                
                self.measurements_text.set_text(measurements_str)
                
                # Update status
                status_str = "System Status:\n\n"
                status_str += "🟢 Audio Input: Active\n"
                status_str += "🟢 Signal Processing: OK\n"
                status_str += "🟢 FFT Analysis: OK\n"
                status_str += "🟢 Measurements: OK\n"
                status_str += f"📊 Sample Rate: {self.sample_rate} Hz\n"
                status_str += f"📈 Buffer Size: {len(self.latest_data)}\n"
                status_str += f"🔧 FFT Size: {self.fft_size}\n"
                
                self.status_text.set_text(status_str)
                
            else:
                # Show error
                error_str = f"❌ Processing Error:\n{result.get('error', 'Unknown error')}"
                self.measurements_text.set_text(error_str)
                
                status_str = "System Status:\n\n"
                status_str += "🔴 Signal Processing: Error\n"
                status_str += f"❌ Error: {result.get('error', 'Unknown')}"
                
                self.status_text.set_text(status_str)
            
        except Exception as e:
            error_str = f"❌ Plot Update Error:\n{str(e)}"
            self.measurements_text.set_text(error_str)
        
        return self.line_time, self.line_freq
    
    def run(self):
        """Main execution function."""
        print("🎤 Simplified Microphone Oscilloscope Test")
        print("=" * 50)
        print("✅ No MCP dependency conflicts")
        print("🔧 Core signal processing algorithms")
        print("📊 Real-time FFT and measurements")
        
        # Select audio device
        device_id = self.select_audio_device()
        
        print(f"\n🎵 Starting audio capture...")
        print("💡 Speak into your microphone or play some audio")
        print("📊 Real-time signal processing active")
        print("🔄 Close the plot window to stop")
        
        try:
            # Start audio stream
            with sd.InputStream(
                device=device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                dtype=self.dtype,
                callback=self.audio_callback
            ):
                self.is_recording = True
                
                # Start animation
                ani = animation.FuncAnimation(
                    self.fig, 
                    self.update_plots, 
                    interval=50,  # 50ms updates = 20 FPS
                    blit=False,
                    cache_frame_data=False
                )
                
                # Show plot
                plt.show()
                
        except Exception as e:
            print(f"❌ Audio capture error: {e}")
        
        finally:
            self.is_recording = False
            print("\n🛑 Audio capture stopped")


def main():
    """Main entry point."""
    print("Simplified Microphone Oscilloscope Test")
    print("Avoiding typer dependency conflicts")
    print("-" * 50)
    
    # Check dependencies
    missing_deps = []
    try:
        import sounddevice as sd
    except ImportError:
        missing_deps.append("sounddevice")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        from scipy import signal
    except ImportError:
        missing_deps.append("scipy")
    
    if missing_deps:
        print(f"❌ Missing required packages: {', '.join(missing_deps)}")
        print("\nTo install required packages:")
        print(f"pip install {' '.join(missing_deps)}")
        return
    
    print("✅ All dependencies available")
    
    # Run the oscilloscope
    oscilloscope = SimpleMicrophoneOscilloscope()
    
    try:
        oscilloscope.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()