# Testing Guide: Oscilloscope MCP Server with Microphone Input

## Overview
This guide provides step-by-step testing instructions for the Oscilloscope MCP Server with microphone input on Windows using Claude Desktop.

## Prerequisites
- Completed Windows setup (see WINDOWS_SETUP.md)
- Claude Desktop running with MCP server configured
- Working microphone

## Test Scenarios

### 1. Basic Functionality Test

**Objective:** Verify the MCP server is running and accessible from Claude Desktop.

**Steps:**
1. Start the MCP server using `start-mcp-server.ps1`
2. Open Claude Desktop
3. In Claude Desktop, ask: "Can you check the hardware status of the oscilloscope server?"

**Expected Result:**
```json
{
  "status": "success",
  "current_configuration": {
    "hardware_interface": "microphone",
    "audio_sample_rate": 44100,
    "channels": 4,
    "microphone_device": "default"
  }
}
```

### 2. Device Discovery Test

**Objective:** Verify audio device enumeration works correctly.

**Steps:**
1. In Claude Desktop, ask: "Please list the available audio devices"
2. Review the returned device list

**Expected Result:**
```json
{
  "status": "success",
  "available_devices": ["default", "microphone", "line-in", "usb-audio"],
  "current_device": "default"
}
```

### 3. Microphone Audio Capture Test

**Objective:** Test real-time audio capture from microphone.

**Steps:**
1. Ensure your microphone is working and not muted
2. In Claude Desktop, ask: "Can you capture 3 seconds of audio from my microphone?"
3. Speak or make noise into the microphone during capture

**Expected Result:**
```json
{
  "status": "success",
  "acquisition_id": "acq_1234567890_abcdef123",
  "hardware_interface": "microphone",
  "sample_rate": 44100,
  "channels_available": [0, 1, 2, 3]
}
```

### 4. Signal Analysis Test

**Objective:** Test signal parameter measurement on captured audio.

**Steps:**
1. First capture audio: "Capture 5 seconds of audio from my microphone"
2. Note the acquisition_id from the response
3. Request analysis: "Please analyze the signal parameters for acquisition [ID], measuring frequency, amplitude, and RMS"

**Expected Result:**
```json
{
  "status": "success",
  "results": {
    "frequency": 1250.5,
    "amplitude": 0.75,
    "rms": 0.53
  }
}
```

### 5. Spectrum Analysis Test

**Objective:** Test FFT spectrum analysis on captured audio.

**Steps:**
1. Play a tone (use phone app or online tone generator)
2. Capture audio: "Capture 3 seconds of audio while I play a 1000Hz tone"
3. Analyze spectrum: "Please analyze the frequency spectrum of acquisition [ID] using hamming window"

**Expected Result:**
```json
{
  "status": "success",
  "spectrum_data": {
    "frequencies": [0, 100, 200, ...],
    "magnitude": [10.5, 25.3, 150.7, ...],
    "window_used": "hamming"
  }
}
```

### 6. Device Configuration Test

**Objective:** Test dynamic device configuration changes.

**Steps:**
1. Check current config: "What's the current hardware status?"
2. Change sample rate: "Please configure the hardware for microphone input at 48000 Hz sample rate"
3. Verify change: "Check the hardware status again"

**Expected Result:**
```json
{
  "status": "success",
  "message": "Hardware reconfigured successfully",
  "configuration": {
    "hardware_interface": "microphone",
    "audio_sample_rate": 48000
  }
}
```

### 7. Test Signal Generation Test

**Objective:** Test synthetic signal generation for calibration.

**Steps:**
1. Generate test signal: "Generate a 1000Hz sine wave test signal with 1.0 amplitude for 2 seconds"
2. Analyze generated signal: "Measure the frequency and amplitude of the generated signal"

**Expected Result:**
```json
{
  "status": "success",
  "signal_type": "sine",
  "frequency": 1000,
  "amplitude": 1.0,
  "samples_generated": 88200
}
```

### 8. Protocol Decoding Test

**Objective:** Test digital protocol decoding capabilities.

**Steps:**
1. Capture audio with digital signal (or use test signal)
2. Decode protocol: "Please decode UART protocol from acquisition [ID] using channels [0] with baud rate 9600"

**Expected Result:**
```json
{
  "status": "success",
  "protocol": "uart",
  "decoded_data": {
    "decoded_data": [
      {"timestamp": 0.001, "type": "start", "data": null},
      {"timestamp": 0.002, "type": "data", "data": "0x42"}
    ]
  }
}
```

### 9. Resource Access Test

**Objective:** Test MCP resource endpoints.

**Steps:**
1. Access acquisition status: "Can you get the latest acquisition status resource?"
2. Access device list: "Please get the hardware device list resource"

**Expected Result:**
Status resource should show current acquisitions and hardware state.

### 10. Error Handling Test

**Objective:** Test error handling and recovery.

**Steps:**
1. Try invalid acquisition ID: "Analyze signal parameters for acquisition 'invalid_id'"
2. Try invalid device: "Configure hardware with microphone_device 'nonexistent'"

**Expected Result:**
```json
{
  "status": "error",
  "error": "Acquisition invalid_id not found"
}
```

## Interactive Testing Scenarios

### Scenario A: Music Analysis
1. Play music through speakers
2. Capture audio with microphone
3. Analyze frequency spectrum
4. Identify dominant frequencies

### Scenario B: Voice Analysis
1. Record yourself speaking
2. Measure voice characteristics (frequency, amplitude)
3. Analyze speech patterns

### Scenario C: Environmental Sound
1. Capture ambient room noise
2. Analyze noise characteristics
3. Measure sound levels

### Scenario D: Function Generator Input
1. Connect external function generator to microphone input
2. Generate known test signals
3. Verify measurement accuracy

## Troubleshooting Tests

### Debug Mode Test
1. Enable debug: "Configure hardware with debug=true"
2. Capture audio and check console output
3. Verify detailed logging appears

### Device Switching Test
1. List available devices
2. Switch to different device
3. Test audio capture with new device

### Sample Rate Test
1. Test different sample rates (22050, 44100, 48000)
2. Verify audio quality and capture success
3. Compare analysis results

## Performance Tests

### Latency Test
1. Measure time from capture request to data available
2. Should be under 3 seconds for 2-second capture

### Concurrent Operations Test
1. Start audio capture
2. Simultaneously request device list
3. Verify both operations complete successfully

### Memory Usage Test
1. Capture multiple long audio samples
2. Monitor memory usage
3. Verify no memory leaks

## Validation Criteria

### ✅ Pass Criteria
- All API calls return proper JSON responses
- Audio capture produces reasonable signal data
- Spectrum analysis shows expected frequency content
- Device configuration changes work
- Error handling provides clear messages

### ❌ Fail Criteria
- API calls return errors or malformed responses
- Audio capture fails or produces no data
- Spectrum analysis shows no signal content
- Configuration changes don't take effect
- Server crashes or becomes unresponsive

## Advanced Testing

### Multi-Channel Test
1. Configure for multiple channels
2. Verify all channels receive data
3. Test channel-specific analysis

### High Sample Rate Test
1. Configure for 96kHz sample rate
2. Test capture and analysis performance
3. Verify frequency resolution improvement

### Long Duration Test
1. Capture audio for 30+ seconds
2. Verify memory usage remains stable
3. Test analysis on large datasets

## Automation Testing

### Batch Test Script
Create a test script that:
1. Starts the server
2. Runs through all test scenarios
3. Logs results
4. Reports pass/fail status

### Continuous Integration
Set up automated testing that:
1. Tests on different Windows versions
2. Validates with different audio devices
3. Checks for regressions

## Reporting Issues

When reporting issues, include:
1. Windows version and configuration
2. Audio device details
3. Console output with debug enabled
4. Specific error messages
5. Steps to reproduce

## Test Environment Setup

### Recommended Test Equipment
- USB microphone for consistent results
- Function generator for known signals
- Audio interface for professional testing
- Multiple audio sources for variety

### Test Data Collection
- Save test results for comparison
- Document performance metrics
- Track success rates across devices
- Monitor system resource usage

## Next Steps After Testing
1. Document any device-specific configurations
2. Create custom workflows for your use case
3. Explore advanced analysis features
4. Integrate with other audio processing tools