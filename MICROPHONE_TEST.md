# Microphone Oscilloscope Test

This script demonstrates real hardware integration by capturing audio from your microphone and processing it through the MCP oscilloscope server components.

## 🎯 What This Tests

✅ **Real Audio Input**: Captures live microphone data  
✅ **MCP Signal Processing**: Processes audio through MCP server  
✅ **Real-time Analysis**: FFT, measurements, and visualization  
✅ **Hardware Integration**: Tests the complete signal chain  

## 🔧 Windows Installation & Setup

### 1. Install Audio Dependencies
```powershell
# Open PowerShell in the project directory
cd "C:\Users\Manas Pandey\Documents\github\Oscilloscope_MCP"

# Install required audio packages
pip install sounddevice matplotlib numpy

# Alternative if above fails:
pip install --user sounddevice matplotlib numpy
```

### 2. Install Audio Drivers (if needed)
If you get audio errors, install Windows audio drivers:
- **ASIO drivers** for professional audio interfaces
- **Windows Audio** should work for built-in microphones

### 3. Test Audio Setup
```powershell
# Test if Python can access your microphone
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## 🚀 Running the Test

### Method 1: Direct Python
```powershell
# Navigate to project directory (use quotes for spaces!)
cd "C:\Users\Manas Pandey\Documents\github\Oscilloscope_MCP"

# Run the microphone test
python mic_oscilloscope_test.py
```

### Method 2: With Virtual Environment
```powershell
# Create and activate virtual environment
python -m venv mic_test_env
.\mic_test_env\Scripts\Activate.ps1

# Install dependencies
pip install sounddevice matplotlib numpy

# Add project to Python path and run
$env:PYTHONPATH = "src"
python mic_oscilloscope_test.py
```

### Method 3: Using Conda (Alternative)
```powershell
# Create conda environment
conda create -n mic_osc python=3.9
conda activate mic_osc

# Install packages
conda install numpy matplotlib
pip install sounddevice

# Run test
cd "C:\Users\Manas Pandey\Documents\github\Oscilloscope_MCP"
python mic_oscilloscope_test.py
```

## 🎤 Using the Test

### 1. Device Selection
When you run the script, it will show available audio devices:
```
🎤 Available Audio Input Devices:
  0: Microphone (Realtek Audio)
  1: Line In (Realtek Audio)
  2: USB Headset Microphone

Select device (0-2, or press Enter for default):
```

### 2. Real-time Display
The script opens a window with 4 panels:

**🌊 Time Domain Waveform**
- Shows real-time audio signal
- X-axis: Time (seconds)
- Y-axis: Amplitude

**📊 Frequency Spectrum (FFT)**
- Shows frequency content of audio
- X-axis: Frequency (Hz, up to 8kHz)
- Y-axis: Magnitude (dB)

**📏 Real-time Measurements**
- RMS value
- Peak-to-peak amplitude
- Dominant frequency
- Maximum amplitude

**⚙️ System Status**
- MCP server status
- Signal processing status
- Sample rate and buffer info

### 3. Testing Ideas

**🎵 Audio Tests:**
- **Speak into microphone** - See voice frequencies (100-3000 Hz)
- **Whistle** - See pure tone spikes
- **Play music** - See complex frequency content
- **Tap microphone** - See impulse response

**🔧 Technical Tests:**
- **Frequency sweep** - Play sine wave sweep to test frequency response
- **White noise** - Test broadband processing
- **Silence** - Check noise floor and baseline

## 🎛️ What You'll See

### Normal Operation
```
✅ MCP components initialized successfully
🎵 Starting audio capture...
💡 Speak into your microphone or play some audio
📊 Real-time processing through MCP server components
```

### Live Measurements Example
```
MCP Server Measurements:

RMS: 0.0234
Peak-to-Peak: 0.1456
Frequency: 440.2 Hz
Amplitude: 0.0892
```

### System Status
```
System Status:

🟢 MCP Server: Active
🟢 Signal Processing: OK
🟢 Measurements: OK
📊 Sample Rate: 44100 Hz
📈 Buffer Size: 1024
```

## 🔧 Troubleshooting

### ❌ "No module named sounddevice"
```powershell
pip install sounddevice
# Or try:
conda install -c conda-forge python-sounddevice
```

### ❌ "No audio devices found"
- Check microphone is connected and enabled
- Try different audio device in Windows settings
- Install audio drivers

### ❌ "PortAudio library not found"
```powershell
# Install PortAudio manually
pip install --upgrade sounddevice
# Or use conda:
conda install portaudio
```

### ❌ "Permission denied" (microphone access)
- Check Windows Privacy Settings
- Allow microphone access for Python/Terminal
- Run PowerShell as Administrator

### ❌ "Cannot import MCP modules"
```powershell
# Make sure you're in the right directory
cd "C:\Users\Manas Pandey\Documents\github\Oscilloscope_MCP"

# Set Python path
$env:PYTHONPATH = "src"
python mic_oscilloscope_test.py
```

## 🎯 Success Criteria

✅ **Working Test Shows:**
1. Audio waveform updating in real-time
2. FFT spectrum showing frequency content
3. Measurements updating with audio changes
4. No error messages in MCP processing
5. Smooth real-time performance

✅ **This Proves:**
- MCP server components work correctly
- Signal processing algorithms function
- Real-time data flow is working
- Hardware integration path is ready

## 📊 Performance Notes

- **Update Rate**: 20 FPS (50ms intervals)
- **Audio Latency**: ~50-100ms (normal for real-time processing)
- **Sample Rate**: 44.1 kHz (CD quality)
- **Buffer Size**: 1024 samples (~23ms at 44.1 kHz)

## 🔄 Next Steps

After successful microphone testing:

1. **Real ADC Integration**: Replace microphone input with ADC data
2. **Multiple Channels**: Extend to multi-channel inputs
3. **Higher Sample Rates**: Test with faster ADCs
4. **Custom Triggers**: Implement hardware trigger conditions
5. **Data Logging**: Save measurements to files

This test validates that your MCP oscilloscope server can handle real-time signal processing with actual hardware inputs!