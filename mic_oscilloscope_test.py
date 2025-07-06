#!/usr/bin/env python3
"""
Microphone Oscilloscope Test Script

This script captures audio from your microphone and processes it through the 
MCP oscilloscope server to test hardware integration and signal processing.

Requirements:
- sounddevice: pip install sounddevice
- numpy: pip install numpy
- matplotlib: pip install matplotlib

Usage:
    python mic_oscilloscope_test.py
"""

import sys
import time
import threading
import queue
import numpy as np
import json
from typing import Dict, Any, List

try:
    import sounddevice as sd
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nInstall required packages:")
    print("pip install sounddevice matplotlib numpy")
    sys.exit(1)

# Add src to path for MCP server imports
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from oscilloscope_mcp.hardware.simulation import SimulatedHardware
    from oscilloscope_mcp.signal_processing.engine import SignalProcessor
    from oscilloscope_mcp.measurements.analyzer import MeasurementAnalyzer
except ImportError as e:
    print(f"❌ Cannot import MCP modules: {e}")
    print("Make sure you're running this script from the Oscilloscope_MCP directory")
    sys.exit(1)


class MicrophoneOscilloscope:
    """Real-time microphone oscilloscope using MCP server components."""
    
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
        
        # MCP Components
        self.hardware = None
        self.signal_processor = None
        self.measurement_analyzer = None
        
        # Real-time plotting
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Microphone Oscilloscope - MCP Server Test', fontsize=14, fontweight='bold')
        
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
        self.ax_measurements.set_title('Real-time Measurements')
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
    
    async def initialize_mcp_components(self):
        """Initialize MCP server components."""
        print("🔧 Initializing MCP server components...")
        
        try:
            # Initialize hardware (simulation mode)
            self.hardware = SimulatedHardware()
            config = {
                "hardware_interface": "simulation",
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "buffer_size": self.buffer_size
            }
            await self.hardware.initialize(config)
            
            # Initialize signal processor
            self.signal_processor = SignalProcessor()
            await self.signal_processor.initialize()
            
            # Initialize measurement analyzer
            self.measurement_analyzer = MeasurementAnalyzer()
            await self.measurement_analyzer.initialize()
            
            print("✅ MCP components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize MCP components: {e}")
            return False
    
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
    
    async def process_audio_with_mcp(self, audio_data):
        """Process audio data through MCP server components."""
        try:
            # Create mock waveform data structure for MCP processing
            time_axis = np.linspace(0, len(audio_data) / self.sample_rate, len(audio_data))
            
            mock_waveform = {
                "waveform_data": {
                    "time": time_axis.tolist(),
                    "channels": {
                        0: {
                            "voltage": audio_data.tolist(),
                            "voltage_range": 1.0,
                            "coupling": "AC",  # Audio is AC coupled
                            "impedance": "1M"
                        }
                    }
                },
                "metadata": {
                    "sample_rate": self.sample_rate,
                    "timestamp": time.time()
                }
            }
            
            # Process through signal processor
            processed = await self.signal_processor.process_waveform(mock_waveform)
            
            # Perform FFT analysis
            spectrum = await self.signal_processor.analyze_spectrum(processed, window="hamming")
            
            # Perform measurements
            measurements = await self.measurement_analyzer.measure_parameters(
                processed, 
                ["rms", "peak_to_peak", "frequency", "amplitude"],
                statistics=False
            )
            
            return {
                "processed_waveform": processed,
                "spectrum": spectrum,
                "measurements": measurements,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def update_plots(self, frame):
        """Update plots with latest data."""
        try:
            # Get latest audio data
            if not self.audio_queue.empty():
                self.latest_data = self.audio_queue.get_nowait()
            
            if len(self.latest_data) == 0:
                return self.line_time, self.line_freq
            
            # Process audio through MCP server (synchronous call for real-time)
            import asyncio
            
            # Create event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Process audio
            result = loop.run_until_complete(self.process_audio_with_mcp(self.latest_data))
            
            if result["success"]:
                # Update time domain plot
                time_axis = np.linspace(0, len(self.latest_data) / self.sample_rate, len(self.latest_data))
                self.line_time.set_data(time_axis, self.latest_data)
                self.ax_time.set_xlim(0, max(time_axis))
                self.ax_time.set_ylim(-1.0, 1.0)
                
                # Update frequency domain plot
                spectrum = result["spectrum"]
                if "frequency" in spectrum and "magnitude_db" in spectrum:
                    freq_data = np.array(spectrum["frequency"])
                    mag_data = np.array(spectrum["magnitude_db"])
                    
                    # Only show up to 8 kHz for audio
                    max_freq_idx = np.searchsorted(freq_data, 8000)
                    freq_data = freq_data[:max_freq_idx]
                    mag_data = mag_data[:max_freq_idx]
                    
                    self.line_freq.set_data(freq_data, mag_data)
                    self.ax_freq.set_xlim(0, 8000)
                    self.ax_freq.set_ylim(-80, 20)
                
                # Update measurements
                measurements = result["measurements"]
                measurements_str = "MCP Server Measurements:\n\n"
                measurements_str += f"RMS: {measurements.get('rms', 0):.4f}\n"
                measurements_str += f"Peak-to-Peak: {measurements.get('peak_to_peak', 0):.4f}\n"
                measurements_str += f"Frequency: {measurements.get('frequency', 0):.1f} Hz\n"
                measurements_str += f"Amplitude: {measurements.get('amplitude', 0):.4f}\n"
                
                self.measurements_text.set_text(measurements_str)
                
                # Update status
                status_str = "System Status:\n\n"
                status_str += "🟢 MCP Server: Active\n"
                status_str += "🟢 Signal Processing: OK\n"
                status_str += "🟢 Measurements: OK\n"
                status_str += f"📊 Sample Rate: {self.sample_rate} Hz\n"
                status_str += f"📈 Buffer Size: {len(self.latest_data)}\n"
                
                self.status_text.set_text(status_str)
                
            else:
                # Show error
                error_str = f"❌ MCP Error:\n{result.get('error', 'Unknown error')}"
                self.measurements_text.set_text(error_str)
                
                status_str = "System Status:\n\n"
                status_str += "🔴 MCP Server: Error\n"
                status_str += "🔴 Signal Processing: Failed\n"
                
                self.status_text.set_text(status_str)
            
        except Exception as e:
            error_str = f"❌ Plot Update Error:\n{str(e)}"
            self.measurements_text.set_text(error_str)
        
        return self.line_time, self.line_freq
    
    async def run(self):
        """Main execution function."""
        print("🎤 Microphone Oscilloscope Test - MCP Server Integration")
        print("=" * 60)
        
        # Initialize MCP components
        if not await self.initialize_mcp_components():
            return
        
        # Select audio device
        device_id = self.select_audio_device()
        
        print(f"\n🎵 Starting audio capture...")
        print("💡 Speak into your microphone or play some audio")
        print("📊 Real-time processing through MCP server components")
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
            
            # Cleanup MCP components
            if self.hardware:
                await self.hardware.cleanup()


def main():
    """Main entry point."""
    print("Microphone Oscilloscope Test Script")
    print("Testing MCP Server Hardware Integration")
    print("-" * 50)
    
    # Check dependencies
    try:
        import sounddevice as sd
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("\nTo install required packages:")
        print("pip install sounddevice matplotlib numpy")
        return
    
    # Run the oscilloscope
    oscilloscope = MicrophoneOscilloscope()
    
    import asyncio
    try:
        asyncio.run(oscilloscope.run())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()