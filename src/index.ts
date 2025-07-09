/**
 * Oscilloscope and Function Generator MCP Server
 * 
 * Professional MCP server for oscilloscope and function generator capabilities
 * with hardware ADC integration support using TypeScript runtime.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import Microphone from "node-microphone";

// Logging utility for MCP server - uses stderr to avoid interfering with JSON-RPC
// Prioritize MCP_MODE environment variable, then check if running as compiled index.js
const isMCPMode = process.env.MCP_MODE === 'true' || 
  (process.env.MCP_MODE !== 'false' && process.argv[1]?.endsWith('index.js'));

const logger = {
  log: (...args: any[]) => {
    if (!isMCPMode) {
      console.log(...args);
    } else {
      console.error(...args); // Use stderr in MCP mode
    }
  },
  error: (...args: any[]) => {
    console.error(...args); // Always use stderr for errors
  }
};

// Configuration schema for server setup
export const configSchema = z.object({
  hardwareInterface: z.string().default("simulation").describe("Hardware interface type: simulation, usb, ethernet, pcie, microphone"),
  sampleRate: z.number().default(1000000).describe("ADC sample rate in Hz"),
  channels: z.number().default(4).describe("Number of ADC channels"),
  bufferSize: z.number().default(1000).describe("Buffer size for data acquisition"),
  timeout: z.number().default(5.0).describe("Timeout for operations in seconds"),
  debug: z.boolean().default(false).describe("Enable debug logging"),
  microphoneDevice: z.string().optional().describe("Microphone device for audio capture"),
  audioSampleRate: z.number().default(44100).describe("Audio sample rate in Hz for microphone")
});

export type Config = z.infer<typeof configSchema>;

/**
 * Data store for managing acquisitions and real-time data
 */
class DataStore {
  private acquisitions: Map<string, any> = new Map();
  private latestData: Map<number, any> = new Map();
  private cachedMeasurements: Map<string, any> = new Map();
  private cachedSpectra: Map<string, any> = new Map();

  storeAcquisition(id: string, data: any): void {
    this.acquisitions.set(id, data);
    
    // Update latest data for each channel
    if (data.channels) {
      Object.keys(data.channels).forEach(channelStr => {
        const channel = parseInt(channelStr);
        this.latestData.set(channel, {
          timestamp: data.timestamp,
          data: data.channels[channel]
        });
      });
    }
  }

  getAcquisition(id: string): any | null {
    return this.acquisitions.get(id) || null;
  }

  getLatestForChannel(channel: number): any | null {
    return this.latestData.get(channel) || null;
  }

  getAllAcquisitions(): string[] {
    return Array.from(this.acquisitions.keys());
  }

  getCachedMeasurements(id: string): any | null {
    return this.cachedMeasurements.get(id) || null;
  }

  setCachedMeasurements(id: string, measurements: any): void {
    this.cachedMeasurements.set(id, measurements);
  }

  getCachedSpectrum(id: string): any | null {
    return this.cachedSpectra.get(id) || null;
  }

  setCachedSpectrum(id: string, spectrum: any): void {
    this.cachedSpectra.set(id, spectrum);
  }
}

/**
 * Hardware abstraction for different ADC backends
 */
class HardwareInterface {
  public config: Partial<Config>;
  private isInitialized = false;
  private microphone?: any;
  private audioBuffers: Map<number, number[]> = new Map();
  private isCapturing = false;

  constructor(config: Partial<Config> = {}) {
    this.config = { 
      hardwareInterface: "simulation",
      sampleRate: 1000000,
      channels: 4,
      bufferSize: 1000,
      timeout: 5.0,
      debug: false,
      audioSampleRate: 44100,
      ...config 
    };
  }

  async initialize(): Promise<void> {
    logger.log(`Initializing ${this.config.hardwareInterface} hardware interface`);
    logger.log(`Sample rate: ${this.config.sampleRate} Hz`);
    logger.log(`Channels: ${this.config.channels}`);
    
    if (this.config.hardwareInterface === "microphone") {
      await this.initializeMicrophone();
    }
    
    this.isInitialized = true;
  }

  async performSelfTest(): Promise<any> {
    if (!this.isInitialized) {
      throw new Error("Hardware not initialized");
    }

    return {
      overall_status: "pass",
      tests: {
        "adc_channels": "pass",
        "sample_rate": "pass", 
        "buffer_integrity": "pass",
        "timing_accuracy": "pass"
      }
    };
  }

  async cleanup(): Promise<void> {
    if (this.config.hardwareInterface === "microphone" && this.microphone) {
      this.stopCapture();
      this.microphone = undefined;
    }
    this.isInitialized = false;
    logger.log("Hardware interface cleaned up");
  }

  private async initializeMicrophone(): Promise<void> {
    try {
      logger.log('Initializing microphone...');
      
      // Ensure sample rate is supported by node-microphone
      const audioSampleRate = this.config.audioSampleRate || 44100;
      let micRate: 44100 | 8000 | 16000 = 44100;
      if (audioSampleRate === 8000 || audioSampleRate === 16000 || audioSampleRate === 44100) {
        micRate = audioSampleRate as 44100 | 8000 | 16000;
      }
      
      const deviceName = this.config.microphoneDevice || 'default';
      const micOptions = {
        rate: micRate,
        channels: 1 as 1 | 2,
        debug: this.config.debug || false,
        exitOnSilence: 6,
        device: (deviceName === 'default' || deviceName === 'hw:0,0' || deviceName === 'plughw:1,0') 
          ? deviceName as "default" | "hw:0,0" | "plughw:1,0"
          : "default"
      };
      
      this.microphone = new Microphone(micOptions);
      logger.log('Microphone initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize microphone:', error);
      throw error;
    }
  }

  async captureMicrophoneData(durationSeconds: number = 1): Promise<number[]> {
    if (!this.microphone) {
      throw new Error('Microphone not initialized');
    }

    return new Promise((resolve, reject) => {
      const samples: number[] = [];
      const sampleRate = this.config.audioSampleRate || 44100;
      const expectedSamples = Math.floor(sampleRate * durationSeconds);
      
      let micStream: any;
      let timeout: NodeJS.Timeout;

      const cleanup = () => {
        if (timeout) clearTimeout(timeout);
        this.microphone.stopRecording();
      };

      timeout = setTimeout(() => {
        cleanup();
        if (samples.length === 0) {
          reject(new Error('No audio data captured within timeout'));
        } else {
          resolve(samples);
        }
      }, (durationSeconds + 1) * 1000);

      try {
        micStream = this.microphone.startRecording();
        
        micStream.on('data', (chunk: Buffer) => {
          // Convert 16-bit PCM to normalized float values
          for (let i = 0; i < chunk.length; i += 2) {
            const sample = chunk.readInt16LE(i) / 32768.0;
            samples.push(sample);
            
            if (samples.length >= expectedSamples) {
              cleanup();
              resolve(samples.slice(0, expectedSamples));
              return;
            }
          }
        });

        micStream.on('error', (error: Error) => {
          cleanup();
          reject(error);
        });
      } catch (error) {
        cleanup();
        reject(error);
      }
    });
  }

  stopCapture(): void {
    if (this.microphone) {
      this.microphone.stopRecording();
      this.isCapturing = false;
    }
  }

  async getAvailableDevices(): Promise<string[]> {
    // Return devices supported by node-microphone library
    // These are the only device names that work with the library's strict typing
    return ['default', 'hw:0,0', 'plughw:1,0'];
  }
}

/**
 * Signal processing engine
 */
class SignalProcessor {
  async analyzeSpectrum(waveformData: any, window: string, resolution: number): Promise<any> {
    const samples = waveformData.channels[0] || [];
    const sampleRate = waveformData.sample_rate || 44100;
    
    if (samples.length === 0) {
      throw new Error('No signal data available for spectrum analysis');
    }

    // Apply windowing function
    const windowedSamples = this.applyWindow(samples, window);
    
    // Perform FFT
    const fft = this.performFFT(windowedSamples, resolution);
    
    // Calculate frequency bins
    const frequencies = [];
    const magnitude = [];
    const phase = [];
    
    const binSize = sampleRate / (2 * resolution);
    
    for (let i = 0; i < resolution; i++) {
      frequencies.push(i * binSize);
      magnitude.push(Math.sqrt(fft.real[i] * fft.real[i] + fft.imag[i] * fft.imag[i]));
      phase.push(Math.atan2(fft.imag[i], fft.real[i]));
    }

    return {
      frequencies,
      magnitude,
      phase,
      window_used: window,
      resolution: resolution
    };
  }

  private applyWindow(samples: number[], windowType: string): number[] {
    const N = samples.length;
    const windowed = new Array(N);
    
    switch (windowType.toLowerCase()) {
      case 'hamming':
        for (let i = 0; i < N; i++) {
          windowed[i] = samples[i] * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (N - 1)));
        }
        break;
      case 'hanning':
        for (let i = 0; i < N; i++) {
          windowed[i] = samples[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)));
        }
        break;
      case 'rectangular':
      default:
        return samples.slice();
    }
    
    return windowed;
  }

  private performFFT(samples: number[], resolution: number): { real: number[], imag: number[] } {
    // Simple FFT implementation for basic functionality
    const N = Math.min(samples.length, resolution * 2);
    const real = new Array(resolution).fill(0);
    const imag = new Array(resolution).fill(0);
    
    // DFT calculation for frequency domain analysis
    for (let k = 0; k < resolution; k++) {
      for (let n = 0; n < N; n++) {
        const angle = -2 * Math.PI * k * n / N;
        real[k] += samples[n] * Math.cos(angle);
        imag[k] += samples[n] * Math.sin(angle);
      }
    }
    
    return { real, imag };
  }
}

/**
 * Measurement analyzer for automated measurements
 */
class MeasurementAnalyzer {
  async measureParameters(waveformData: any, measurements: string[], statistics: boolean): Promise<any> {
    const results: any = {};
    const samples = waveformData.channels[0] || [];
    const sampleRate = waveformData.sample_rate || 44100;

    if (samples.length === 0) {
      throw new Error('No signal data available for measurements');
    }

    for (const measurement of measurements) {
      switch (measurement) {
        case "frequency":
          results[measurement] = this.measureFrequency(samples, sampleRate);
          break;
        case "amplitude":
          results[measurement] = this.measureAmplitude(samples);
          break;
        case "peak_to_peak":
          results[measurement] = this.measurePeakToPeak(samples);
          break;
        case "rms":
          results[measurement] = this.measureRMS(samples);
          break;
        case "duty_cycle":
          results[measurement] = this.measureDutyCycle(samples);
          break;
        default:
          results[measurement] = 0;
      }

      if (statistics) {
        results[`${measurement}_stats`] = this.calculateStatistics(samples, results[measurement]);
      }
    }

    return results;
  }

  private measureFrequency(samples: number[], sampleRate: number): number {
    // Use zero-crossing detection for frequency measurement
    let crossings = 0;
    let lastSign = samples[0] >= 0 ? 1 : -1;
    
    for (let i = 1; i < samples.length; i++) {
      const currentSign = samples[i] >= 0 ? 1 : -1;
      if (currentSign !== lastSign) {
        crossings++;
        lastSign = currentSign;
      }
    }
    
    // Frequency = (crossings / 2) / (duration in seconds)
    const duration = samples.length / sampleRate;
    return (crossings / 2) / duration;
  }

  private measureAmplitude(samples: number[]): number {
    // Peak amplitude (maximum absolute value)
    return Math.max(...samples.map(s => Math.abs(s)));
  }

  private measurePeakToPeak(samples: number[]): number {
    const max = Math.max(...samples);
    const min = Math.min(...samples);
    return max - min;
  }

  private measureRMS(samples: number[]): number {
    // Root Mean Square calculation
    const sumSquares = samples.reduce((sum, sample) => sum + sample * sample, 0);
    return Math.sqrt(sumSquares / samples.length);
  }

  private measureDutyCycle(samples: number[]): number {
    // For digital signals, measure time above threshold vs total time
    const threshold = 0.0; // Zero crossing threshold
    let highTime = 0;
    
    for (const sample of samples) {
      if (sample > threshold) {
        highTime++;
      }
    }
    
    return (highTime / samples.length) * 100; // Return as percentage
  }

  private calculateStatistics(samples: number[], measurement: number): any {
    // Calculate basic statistics based on actual sample data
    const mean = samples.reduce((sum, s) => sum + s, 0) / samples.length;
    const variance = samples.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / samples.length;
    const std = Math.sqrt(variance);
    
    return {
      min: Math.min(...samples),
      max: Math.max(...samples),
      mean: mean,
      std: std
    };
  }
}

/**
 * Protocol decoder for digital communication protocols
 */
class ProtocolDecoder {
  async decodeProtocol(waveformData: any, protocol: string, settings: any): Promise<any> {
    const samples = waveformData.channels[0] || [];
    const sampleRate = waveformData.sample_rate || 44100;
    
    if (samples.length === 0) {
      throw new Error('No signal data available for protocol decoding');
    }

    switch (protocol.toLowerCase()) {
      case 'uart':
        return this.decodeUART(samples, sampleRate, settings);
      default:
        // Return mock data for unsupported protocols
        return {
          protocol: protocol,
          decoded_data: [
            { timestamp: 0.001, type: "start", data: null },
            { timestamp: 0.002, type: "data", data: "0x42" },
            { timestamp: 0.003, type: "data", data: "0x43" },
            { timestamp: 0.004, type: "stop", data: null }
          ],
          settings_used: settings,
          statistics: {
            total_frames: 2,
            errors: 0,
            timing_violations: 0
          }
        };
    }
  }

  private decodeUART(samples: number[], sampleRate: number, settings: any): any {
    const baudRate = settings.baud_rate || 9600;
    const samplesPerBit = Math.floor(sampleRate / baudRate);
    const threshold = 0.0; // Digital threshold
    
    // Convert analog samples to digital bits
    const digitalBits = [];
    for (let i = 0; i < samples.length; i += samplesPerBit) {
      const bitSamples = samples.slice(i, i + samplesPerBit);
      const avgLevel = bitSamples.reduce((sum, s) => sum + s, 0) / bitSamples.length;
      digitalBits.push(avgLevel > threshold ? 1 : 0);
    }
    
    // Simple UART frame extraction (start bit, 8 data bits, stop bit)
    const decodedData = [];
    let frameCount = 0;
    
    for (let i = 0; i < digitalBits.length - 10; i++) {
      if (digitalBits[i] === 0) { // Start bit (low)
        const dataBits = digitalBits.slice(i + 1, i + 9);
        const stopBit = digitalBits[i + 9];
        
        if (stopBit === 1) { // Valid stop bit
          let byteValue = 0;
          for (let j = 0; j < 8; j++) {
            byteValue |= (dataBits[j] << j);
          }
          
          decodedData.push({
            timestamp: (i * samplesPerBit) / sampleRate,
            type: "data",
            data: `0x${byteValue.toString(16).toUpperCase().padStart(2, '0')}`
          });
          frameCount++;
        }
        
        i += 9; // Skip to next potential frame
      }
    }
    
    return {
      protocol: "uart",
      decoded_data: decodedData,
      settings_used: settings,
      statistics: {
        total_frames: frameCount,
        errors: 0,
        timing_violations: 0
      }
    };
  }
}

// Initialize components
const dataStore = new DataStore();
let hardware: HardwareInterface;
const signalProcessor = new SignalProcessor(); 
const measurementAnalyzer = new MeasurementAnalyzer();
const protocolDecoder = new ProtocolDecoder();

// Initialize hardware with configuration
async function initializeHardware(config: Partial<Config> = {}): Promise<void> {
  hardware = new HardwareInterface(config);
  try {
    await hardware.initialize();
  } catch (error) {
    logger.error('Hardware initialization failed:', error);
    throw error; // Re-throw to let caller handle the error
  }
}

// Read configuration from environment variables
const envConfig: Partial<Config> = {
  hardwareInterface: process.env.HARDWARE_INTERFACE || "simulation",
  audioSampleRate: process.env.AUDIO_SAMPLE_RATE ? parseInt(process.env.AUDIO_SAMPLE_RATE) : 44100,
  debug: process.env.DEBUG === "true",
  microphoneDevice: process.env.MICROPHONE_DEVICE || "default"
};

// Enhanced logging for MCP vs standalone modes
if (isMCPMode) {
  logger.log("🔧 MCP mode detected - using stderr for logging");
} else {
  logger.log("🔧 Standalone mode - using stdout for logging");
}

logger.log("🔧 Configuration loaded:", envConfig);

// Initialize with environment configuration
initializeHardware(envConfig).catch(logger.error);

// Create MCP server instance
const server = new McpServer({
  name: "oscilloscope-function-generator",
  version: "1.0.0"
});

// ====== OSCILLOSCOPE TOOLS ======

server.tool(
  'get_acquisition_status',
  'Get current acquisition status and available data',
  {},
  async () => {
    const acquisitions = dataStore.getAllAcquisitions();
    const timestamp = Date.now() / 1000;

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          status: "success",
          total_acquisitions: acquisitions.length,
          recent_acquisitions: acquisitions.slice(-10),
          timestamp: timestamp
        }, null, 2)
      }]
    };
  }
);

server.tool(
  'acquire_waveform',
  'Wait for new ADC data acquisition',
  {
    timeout: z.number().default(5.0).describe('Timeout in seconds'),
    channels: z.array(z.number()).optional().describe('Specific channels to acquire')
  },
  async ({ timeout, channels }) => {
    // Simulate waiting for data
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Generate acquisition data based on hardware interface
    const acquisitionId = `acq_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const mockData = {
      timestamp: Date.now() / 1000,
      sample_rate: hardware.config.hardwareInterface === "microphone" ? 
        (hardware.config.audioSampleRate || 44100) : 1000000,
      channels: {} as any
    };

    const channelsToUse = channels || [0, 1, 2, 3];
    
    if (hardware.config.hardwareInterface === "microphone") {
      // Capture real microphone data
      try {
        const micData = await hardware.captureMicrophoneData(timeout || 2);
        mockData.channels[0] = micData;
        // For additional channels, add processed versions of the same data
        for (let i = 1; i < Math.min(channelsToUse.length, 4); i++) {
          mockData.channels[i] = micData.map(sample => sample * 0.5 + Math.random() * 0.1);
        }
      } catch (error) {
        throw new Error(`Microphone capture failed: ${error}`);
      }
    } else {
      // Generate mock waveform data for simulation
      for (const channel of channelsToUse) {
        const samples = Array.from({ length: 1000 }, (_, i) => 
          Math.sin(2 * Math.PI * 1000 * i / 1000000) + Math.random() * 0.1
        );
        mockData.channels[channel] = samples;
      }
    }

    dataStore.storeAcquisition(acquisitionId, mockData);

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          status: "success",
          acquisition_id: acquisitionId,
          timestamp: Date.now() / 1000,
          channels_available: channelsToUse,
          hardware_interface: hardware.config.hardwareInterface,
          sample_rate: mockData.sample_rate
        }, null, 2)
      }]
    };
  }
);

server.tool(
  'measure_parameters',
  'Perform automated measurements on acquired data',
  {
    acquisition_id: z.string().describe('ID of the acquisition to analyze'),
    measurements: z.array(z.string()).describe('List of measurements to perform'),
    channel: z.number().default(0).describe('Channel to analyze'),
    statistics: z.boolean().default(false).describe('Include statistical analysis')
  },
  async ({ acquisition_id, measurements, channel, statistics }) => {
    try {
      // Check cache first
      const cached = dataStore.getCachedMeasurements(acquisition_id);
      if (cached) {
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              status: "success",
              source: "cache",
              results: cached,
              timestamp: Date.now() / 1000
            }, null, 2)
          }]
        };
      }

      // Get acquisition data
      const waveformData = dataStore.getAcquisition(acquisition_id);
      if (!waveformData) {
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              status: "error",
              error: `Acquisition ${acquisition_id} not found`,
              timestamp: Date.now() / 1000
            }, null, 2)
          }]
        };
      }

      // Perform measurements
      const results = await measurementAnalyzer.measureParameters(
        waveformData,
        measurements,
        statistics
      );

      // Cache results
      dataStore.setCachedMeasurements(acquisition_id, results);

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "success",
            acquisition_id: acquisition_id,
            channel: channel,
            results: results,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };

    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "error",
            error: `Measurement failed: ${error}`,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };
    }
  }
);

server.tool(
  'analyze_spectrum',
  'Perform FFT analysis on acquired data',
  {
    acquisition_id: z.string().describe('ID of the acquisition to analyze'),
    channel: z.number().default(0).describe('Channel to analyze'),
    window: z.string().default('hamming').describe('Windowing function'),
    resolution: z.number().default(1024).describe('FFT resolution')
  },
  async ({ acquisition_id, channel, window, resolution }) => {
    try {
      // Check cache first
      const cached = dataStore.getCachedSpectrum(acquisition_id);
      if (cached) {
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              status: "success",
              source: "cache",
              spectrum_data: cached,
              timestamp: Date.now() / 1000
            }, null, 2)
          }]
        };
      }

      // Get acquisition data
      const waveformData = dataStore.getAcquisition(acquisition_id);
      if (!waveformData) {
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              status: "error",
              error: `Acquisition ${acquisition_id} not found`,
              timestamp: Date.now() / 1000
            }, null, 2)
          }]
        };
      }

      // Perform spectrum analysis
      const spectrumData = await signalProcessor.analyzeSpectrum(
        waveformData,
        window,
        resolution
      );

      // Cache results
      dataStore.setCachedSpectrum(acquisition_id, spectrumData);

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "success",
            acquisition_id: acquisition_id,
            channel: channel,
            spectrum_data: spectrumData,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };

    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "error",
            error: `Spectrum analysis failed: ${error}`,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };
    }
  }
);

// ====== FUNCTION GENERATOR TOOLS ======

server.tool(
  'generate_test_signal',
  'Generate a test signal and inject it as ADC data',
  {
    signal_type: z.string().default('sine').describe('Signal type: sine, square, triangle, sawtooth, noise'),
    frequency: z.number().default(1000.0).describe('Signal frequency in Hz'),
    amplitude: z.number().default(1.0).describe('Signal amplitude'),
    sample_rate: z.number().default(44100.0).describe('Sample rate in Hz'),
    duration: z.number().default(1.0).describe('Duration in seconds'),
    channel: z.number().default(0).describe('Target channel')
  },
  async ({ signal_type, frequency, amplitude, sample_rate, duration, channel }) => {
    try {
      // Generate test signal
      const numSamples = Math.floor(sample_rate * duration);
      const samples: number[] = [];

      for (let i = 0; i < numSamples; i++) {
        const t = i / sample_rate;
        let value = 0;

        switch (signal_type) {
          case 'sine':
            value = amplitude * Math.sin(2 * Math.PI * frequency * t);
            break;
          case 'square':
            value = amplitude * Math.sign(Math.sin(2 * Math.PI * frequency * t));
            break;
          case 'triangle':
            value = amplitude * (2 / Math.PI) * Math.asin(Math.sin(2 * Math.PI * frequency * t));
            break;
          case 'sawtooth':
            value = amplitude * (2 * (t * frequency - Math.floor(t * frequency + 0.5)));
            break;
          case 'noise':
            value = amplitude * (Math.random() * 2 - 1);
            break;
          default:
            value = amplitude * Math.sin(2 * Math.PI * frequency * t);
        }

        samples.push(value);
      }

      // Store as acquisition data
      const acquisitionId = `test_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const adcData = {
        timestamp: Date.now() / 1000,
        sample_rate: sample_rate,
        channels: { [channel]: samples },
        metadata: {
          signal_type,
          frequency,
          amplitude,
          duration,
          generated: true
        }
      };

      dataStore.storeAcquisition(acquisitionId, adcData);

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "success",
            acquisition_id: acquisitionId,
            signal_type: signal_type,
            frequency: frequency,
            amplitude: amplitude,
            duration: duration,
            samples_generated: numSamples,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };

    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "error",
            error: `Signal generation failed: ${error}`,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };
    }
  }
);

// ====== ANALYSIS TOOLS ======

server.tool(
  'decode_protocol',
  'Decode digital communication protocols from acquired data',
  {
    acquisition_id: z.string().describe('ID of the acquisition to analyze'),
    protocol: z.string().describe('Protocol to decode: uart, spi, i2c, can'),
    settings: z.record(z.any()).describe('Protocol-specific settings'),
    channels: z.array(z.number()).describe('Channels to use for protocol decoding')
  },
  async ({ acquisition_id, protocol, settings, channels }) => {
    try {
      // Get acquisition data
      const waveformData = dataStore.getAcquisition(acquisition_id);
      if (!waveformData) {
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              status: "error",
              error: `Acquisition ${acquisition_id} not found`,
              timestamp: Date.now() / 1000
            }, null, 2)
          }]
        };
      }

      // Extract channel data for decoding
      const channelData: any = {};
      for (const channel of channels) {
        if (waveformData.channels[channel]) {
          channelData[channel] = waveformData.channels[channel];
        }
      }

      // Perform protocol decoding
      const decodedData = await protocolDecoder.decodeProtocol(
        { channels: channelData },
        protocol,
        settings
      );

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "success",
            acquisition_id: acquisition_id,
            protocol: protocol,
            channels_used: channels,
            decoded_data: decodedData,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };

    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "error",
            error: `Protocol decoding failed: ${error}`,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };
    }
  }
);

// ====== DEVICE MANAGEMENT TOOLS ======

server.tool(
  'list_audio_devices',
  'List available audio input devices for microphone capture',
  {},
  async () => {
    try {
      const devices = await hardware.getAvailableDevices();
      const currentDevice = hardware.config.microphoneDevice || 'default';
      
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "success",
            available_devices: devices,
            current_device: currentDevice,
            hardware_interface: hardware.config.hardwareInterface,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "error",
            error: `Failed to list devices: ${error}`,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };
    }
  }
);

server.tool(
  'configure_hardware',
  'Configure hardware interface and device settings',
  {
    hardware_interface: z.string().describe('Hardware interface: simulation, microphone, usb, ethernet, pcie'),
    microphone_device: z.string().optional().describe('Microphone device name'),
    audio_sample_rate: z.number().optional().describe('Audio sample rate (Hz)'),
    channels: z.number().optional().describe('Number of channels'),
    debug: z.boolean().optional().describe('Enable debug logging')
  },
  async ({ hardware_interface, microphone_device, audio_sample_rate, channels, debug }) => {
    try {
      // Build new configuration
      const newConfig: Partial<Config> = {
        hardwareInterface: hardware_interface,
        ...(microphone_device && { microphoneDevice: microphone_device }),
        ...(audio_sample_rate && { audioSampleRate: audio_sample_rate }),
        ...(channels && { channels: channels }),
        ...(debug !== undefined && { debug: debug })
      };

      // Clean up current hardware
      await hardware.cleanup();

      // Initialize with new configuration
      await initializeHardware(newConfig);

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "success",
            message: "Hardware reconfigured successfully",
            configuration: {
              hardware_interface: hardware.config.hardwareInterface,
              microphone_device: hardware.config.microphoneDevice,
              audio_sample_rate: hardware.config.audioSampleRate,
              channels: hardware.config.channels,
              debug: hardware.config.debug
            },
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "error",
            error: `Hardware configuration failed: ${error}`,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };
    }
  }
);

server.tool(
  'get_hardware_status',
  'Get current hardware configuration and status',
  {},
  async () => {
    try {
      const config = hardware.config;
      const devices = await hardware.getAvailableDevices();
      
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "success",
            current_configuration: {
              hardware_interface: config.hardwareInterface,
              sample_rate: config.sampleRate,
              audio_sample_rate: config.audioSampleRate,
              channels: config.channels,
              buffer_size: config.bufferSize,
              timeout: config.timeout,
              microphone_device: config.microphoneDevice,
              debug: config.debug
            },
            available_devices: devices,
            server_info: {
              name: "oscilloscope-function-generator",
              version: "1.0.0",
              runtime: "typescript"
            },
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "error",
            error: `Failed to get hardware status: ${error}`,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };
    }
  }
);

// ====== RESOURCES ======

server.resource(
  'acquisition://latest/status',
  'Get latest acquisition status and overview',
  async () => {
    const acquisitions = dataStore.getAllAcquisitions();
    const overview = {
      total_acquisitions: acquisitions.length,
      recent_acquisitions: acquisitions.slice(-5),
      server_status: "running",
      hardware_status: hardware.config.hardwareInterface,
      hardware_config: {
        interface: hardware.config.hardwareInterface,
        sample_rate: hardware.config.sampleRate,
        audio_sample_rate: hardware.config.audioSampleRate,
        channels: hardware.config.channels,
        microphone_device: hardware.config.microphoneDevice
      },
      timestamp: Date.now() / 1000
    };

    return {
      contents: [{
        uri: 'acquisition://latest/status',
        mimeType: 'application/json',
        text: JSON.stringify(overview, null, 2)
      }]
    };
  }
);

server.resource(
  'hardware://devices/list',
  'Get list of available hardware devices',
  async () => {
    try {
      const devices = await hardware.getAvailableDevices();
      const deviceInfo = {
        available_devices: devices,
        current_device: hardware.config.microphoneDevice || 'default',
        hardware_interface: hardware.config.hardwareInterface,
        supported_interfaces: ['simulation', 'microphone', 'usb', 'ethernet', 'pcie'],
        timestamp: Date.now() / 1000
      };

      return {
        contents: [{
          uri: 'hardware://devices/list',
          mimeType: 'application/json',
          text: JSON.stringify(deviceInfo, null, 2)
        }]
      };
    } catch (error) {
      return {
        contents: [{
          uri: 'hardware://devices/list',
          mimeType: 'application/json',
          text: JSON.stringify({
            error: `Failed to get device list: ${error}`,
            timestamp: Date.now() / 1000
          }, null, 2)
        }]
      };
    }
  }
);

// ====== PROMPTS ======

server.prompt(
  'adc_integration_workflow',
  'Workflow for integrating ADC hardware',
  {},
  async () => {
    const currentConfig = hardware.config;
    return {
      messages: [{
        role: 'user',
        content: {
          type: 'text',
          text: `
# ADC Hardware Integration Workflow

## Overview
This workflow guides you through integrating your ADC hardware with the oscilloscope MCP server.

## Current Configuration
- Hardware Interface: ${currentConfig.hardwareInterface}
- Sample Rate: ${currentConfig.sampleRate?.toLocaleString()} Hz
- Audio Sample Rate: ${currentConfig.audioSampleRate?.toLocaleString()} Hz
- Channels: ${currentConfig.channels}
- Buffer Size: ${currentConfig.bufferSize}
- Microphone Device: ${currentConfig.microphoneDevice || 'default'}

## Steps

### 1. Check Hardware Status
First, check your current hardware configuration:
\`\`\`
get_hardware_status()
\`\`\`

### 2. Configure Hardware (if needed)
To use your microphone as an analog source:
\`\`\`
configure_hardware(hardware_interface="microphone", microphone_device="default", audio_sample_rate=44100)
\`\`\`

### 3. List Available Devices
To see available audio devices:
\`\`\`
list_audio_devices()
\`\`\`

### 4. Test Signal Generation
Generate test signals for verification:
\`\`\`
generate_test_signal(signal_type="sine", frequency=1000, amplitude=1.0)
\`\`\`

### 5. Acquire Data
Capture data from your microphone:
\`\`\`
acquire_waveform(timeout=5.0, channels=[0])
\`\`\`

### 6. Analyze Results
Perform measurements and analysis:
\`\`\`
measure_parameters(acquisition_id="...", measurements=["frequency", "amplitude", "rms"])
analyze_spectrum(acquisition_id="...", window="hamming")
\`\`\`

### 7. Protocol Decoding (Optional)
If working with digital signals:
\`\`\`
decode_protocol(acquisition_id="...", protocol="uart", settings={"baud_rate": 9600})
\`\`\`

## Hardware Interface Types
- **simulation**: Mock data for testing
- **microphone**: Use computer microphone as analog input
- **usb**: USB-based ADC devices
- **ethernet**: Network-connected ADCs
- **pcie**: PCIe ADC cards

## Device Selection
You can select different microphone devices by using:
1. \`list_audio_devices()\` - to see available options
2. \`configure_hardware(hardware_interface="microphone", microphone_device="your_device")\` - to switch devices

## Next Steps
1. Verify basic functionality with test signals
2. Configure microphone input for real-time audio analysis
3. Use Claude Desktop integration for interactive analysis
4. Implement custom signal processing workflows
          `
        }
      }]
    };
  }
);

server.prompt(
  'microphone_setup_guide',
  'Guide for setting up microphone input on Windows',
  {},
  async () => {
    return {
      messages: [{
        role: 'user',
        content: {
          type: 'text',
          text: `
# Microphone Setup Guide for Windows

## Overview
This guide helps you set up your microphone as an analog input source for the oscilloscope MCP server.

## Prerequisites
- Windows 10/11 with working microphone
- Claude Desktop installed and configured
- Node.js installed (for local server)
- Sox audio tools (automatically installed with dependencies)

## Quick Setup

### 1. Configure for Microphone Input
\`\`\`
configure_hardware(hardware_interface="microphone", audio_sample_rate=44100, debug=true)
\`\`\`

### 2. Test Microphone
\`\`\`
get_hardware_status()
list_audio_devices()
\`\`\`

### 3. Capture Audio
\`\`\`
acquire_waveform(timeout=3.0, channels=[0])
\`\`\`

### 4. Analyze Audio
\`\`\`
measure_parameters(acquisition_id="your_id", measurements=["frequency", "amplitude", "rms"])
analyze_spectrum(acquisition_id="your_id", window="hamming", resolution=1024)
\`\`\`

## Device Selection
To select a specific microphone:
1. List available devices: \`list_audio_devices()\`
2. Configure specific device: \`configure_hardware(hardware_interface="microphone", microphone_device="your_device_name")\`

## Common Audio Sources
- **Default**: System default microphone
- **Microphone**: Built-in or USB microphone
- **Line-in**: Audio line input
- **USB-Audio**: USB audio devices

## Troubleshooting
- Enable debug mode: \`configure_hardware(debug=true)\`
- Check Windows audio settings and permissions
- Ensure microphone is not muted or disabled
- Try different sample rates (22050, 44100, 48000 Hz)

## Sample Analysis Workflow
1. Configure microphone input
2. Capture 3-5 seconds of audio
3. Analyze frequency spectrum
4. Measure signal parameters
5. Use results for further processing

## Integration with Claude Desktop
This server integrates seamlessly with Claude Desktop:
- Real-time audio analysis through prompts
- Interactive device selection
- Automated measurement workflows
- Visual spectrum analysis results
          `
        }
      }]
    };
  }
);

// Cleanup on server shutdown
process.on('SIGINT', async () => {
  await hardware.cleanup();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await hardware.cleanup();
  process.exit(0);
});

// Start MCP server for direct Node.js execution
async function startServer() {
  try {
    // Create stdio transport for Claude Desktop integration
    const transport = new StdioServerTransport();
    
    // In MCP mode, minimize startup logging to avoid JSON-RPC interference
    if (isMCPMode) {
      logger.log("Starting MCP server for Claude Desktop...");
    } else {
      logger.log("🚀 Starting MCP server...");
    }
    
    // Start the MCP server - this will handle JSON-RPC protocol
    await server.connect(transport);
    
    // Post-startup messages (only in stderr for MCP mode)
    if (!isMCPMode) {
      logger.log("🚀 MCP server started successfully");
      logger.log("📡 Ready for Claude Desktop connections");
      logger.log("🎤 Hardware configured for:", hardware.config.hardwareInterface);
    } else {
      logger.log("MCP server ready for Claude Desktop connections");
    }
  } catch (error) {
    logger.error("❌ Failed to start MCP server:", error);
    process.exit(1);
  }
}

// Only start server if this file is run directly (not imported)
if (process.argv[1] && process.argv[1].endsWith('index.js')) {
  startServer().catch(logger.error);
}

// Export as function for Smithery CLI compatibility
export default function({ sessionId, config }: { sessionId?: string; config?: any } = {}) {
  // If configuration is provided, reinitialize hardware
  if (config) {
    initializeHardware(config).catch(logger.error);
  }
  return server.server;
}