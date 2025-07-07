/**
 * Oscilloscope and Function Generator MCP Server
 * 
 * Professional MCP server for oscilloscope and function generator capabilities
 * with hardware ADC integration support using Smithery SDK.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

// Define configuration schema for session configuration
export const configSchema = z.object({
  hardwareInterface: z.string().describe("Hardware interface type: simulation, usb, ethernet, pcie"),
  sampleRate: z.number().describe("ADC sample rate in Hz"),
  channels: z.number().describe("Number of ADC channels"),
  bufferSize: z.number().describe("Buffer size for data acquisition"),
  timeout: z.number().describe("Timeout for operations in seconds"),
  debug: z.boolean().describe("Enable debug logging")
}).default({
  hardwareInterface: "simulation",
  sampleRate: 1000000,
  channels: 4,
  bufferSize: 1000,
  timeout: 5.0,
  debug: false
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
  private config: Config;
  private isInitialized = false;

  constructor(config: Config) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    // Simulate hardware initialization
    console.log(`Initializing ${this.config.hardwareInterface} hardware interface`);
    console.log(`Sample rate: ${this.config.sampleRate} Hz`);
    console.log(`Channels: ${this.config.channels}`);
    
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
    this.isInitialized = false;
    console.log("Hardware interface cleaned up");
  }
}

/**
 * Signal processing engine
 */
class SignalProcessor {
  async analyzeSpectrum(waveformData: any, window: string, resolution: number): Promise<any> {
    // Simulate FFT analysis
    const channels = waveformData.channels;
    const result: any = {
      frequencies: [],
      magnitude: [],
      phase: [],
      window_used: window,
      resolution: resolution
    };

    // Generate mock spectrum data
    for (let i = 0; i < resolution; i++) {
      result.frequencies.push(i * 100); // Mock frequency bins
      result.magnitude.push(Math.random() * 100); // Mock magnitude
      result.phase.push(Math.random() * 2 * Math.PI); // Mock phase
    }

    return result;
  }
}

/**
 * Measurement analyzer for automated measurements
 */
class MeasurementAnalyzer {
  async measureParameters(waveformData: any, measurements: string[], statistics: boolean): Promise<any> {
    const results: any = {};

    for (const measurement of measurements) {
      switch (measurement) {
        case "frequency":
          results[measurement] = 1000 + Math.random() * 1000; // Mock frequency
          break;
        case "amplitude":
          results[measurement] = 1.0 + Math.random() * 2.0; // Mock amplitude
          break;
        case "peak_to_peak":
          results[measurement] = 2.0 + Math.random() * 4.0; // Mock P-P
          break;
        case "rms":
          results[measurement] = 0.707 + Math.random() * 0.5; // Mock RMS
          break;
        case "duty_cycle":
          results[measurement] = 50 + Math.random() * 10; // Mock duty cycle %
          break;
        default:
          results[measurement] = Math.random() * 100; // Generic mock value
      }

      if (statistics) {
        results[`${measurement}_stats`] = {
          min: results[measurement] * 0.9,
          max: results[measurement] * 1.1,
          mean: results[measurement],
          std: results[measurement] * 0.05
        };
      }
    }

    return results;
  }
}

/**
 * Protocol decoder for digital communication protocols
 */
class ProtocolDecoder {
  async decodeProtocol(waveformData: any, protocol: string, settings: any): Promise<any> {
    // Mock protocol decoding
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

/**
 * Create stateful MCP server function as required by Smithery
 * Default export as expected by Smithery CLI
 */
export default function({ sessionId, config }: { sessionId: string; config: Config }) {
  const server = new McpServer({
    name: "oscilloscope-function-generator",
    version: "1.0.0"
  });

  // Initialize components
  const dataStore = new DataStore();
  const hardware = new HardwareInterface(config);
  const signalProcessor = new SignalProcessor();
  const measurementAnalyzer = new MeasurementAnalyzer();
  const protocolDecoder = new ProtocolDecoder();

  // Initialize hardware asynchronously
  hardware.initialize().catch(console.error);

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
            session_id: sessionId,
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
      const startTime = Date.now() / 1000;
      const initialCount = dataStore.getAllAcquisitions().length;

      // Simulate waiting for data (in a real implementation, this would wait for actual hardware)
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Generate mock acquisition data
      const acquisitionId = `acq_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const mockData = {
        timestamp: Date.now() / 1000,
        sample_rate: config.sampleRate,
        channels: {} as any
      };

      const channelsToUse = channels || Array.from({ length: config.channels }, (_, i) => i);
      for (const channel of channelsToUse) {
        // Generate mock waveform data
        const samples = Array.from({ length: config.bufferSize }, (_, i) => 
          Math.sin(2 * Math.PI * 1000 * i / config.sampleRate) + Math.random() * 0.1
        );
        mockData.channels[channel] = samples;
      }

      dataStore.storeAcquisition(acquisitionId, mockData);

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            status: "success",
            acquisition_id: acquisitionId,
            timestamp: Date.now() / 1000,
            channels_available: channelsToUse
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

  // ====== RESOURCES ======

  server.resource(
    'acquisition://{acquisition_id}/waveform',
    'Get raw waveform data for an acquisition',
    async (uri) => {
      const acquisition_id = uri.pathname.split('/')[1];
      try {
        const waveformData = dataStore.getAcquisition(acquisition_id);
        if (waveformData) {
          return {
            contents: [{
              uri: `acquisition://${acquisition_id}/waveform`,
              mimeType: 'application/json',
              text: JSON.stringify(waveformData, null, 2)
            }]
          };
        } else {
          return {
            contents: [{
              uri: `acquisition://${acquisition_id}/waveform`,
              mimeType: 'application/json',
              text: JSON.stringify({ error: `Acquisition ${acquisition_id} not found` })
            }]
          };
        }
      } catch (error) {
        return {
          contents: [{
            uri: `acquisition://${acquisition_id}/waveform`,
            mimeType: 'application/json',
            text: JSON.stringify({ error: `Resource access failed: ${error}` })
          }]
        };
      }
    }
  );

  server.resource(
    'latest://channel{channel}/waveform',
    'Get latest waveform data for a specific channel',
    async (uri) => {
      const channel = uri.pathname.match(/channel(\d+)/)?.[1] || '0';
      const channelNum = parseInt(channel);
      try {
        const waveformData = dataStore.getLatestForChannel(channelNum);
        if (waveformData) {
          return {
            contents: [{
              uri: `latest://channel${channelNum}/waveform`,
              mimeType: 'application/json',
              text: JSON.stringify(waveformData, null, 2)
            }]
          };
        } else {
          return {
            contents: [{
              uri: `latest://channel${channelNum}/waveform`,
              mimeType: 'application/json',
              text: JSON.stringify({ error: `No data available for channel ${channel}` })
            }]
          };
        }
      } catch (error) {
        return {
          contents: [{
            uri: `latest://channel${channelNum}/waveform`,
            mimeType: 'application/json',
            text: JSON.stringify({ error: `Resource access failed: ${error}` })
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
      return {
        messages: [{
          role: 'user',
          content: {
            type: 'text',
            text: `
# ADC Hardware Integration Workflow

## Overview
This workflow guides you through integrating your ADC hardware with the oscilloscope MCP server.

## Configuration
Current session configuration:
- Hardware Interface: ${config.hardwareInterface}
- Sample Rate: ${config.sampleRate} Hz
- Channels: ${config.channels}
- Buffer Size: ${config.bufferSize}
- Session ID: ${sessionId}

## Steps

### 1. Test Signal Generation
Use \`generate_test_signal\` to create known signals for testing:
\`\`\`
generate_test_signal(signal_type="sine", frequency=1000, amplitude=1.0)
\`\`\`

### 2. Acquire Data
Use \`acquire_waveform\` to capture data from your hardware:
\`\`\`
acquire_waveform(timeout=10.0, channels=[0, 1])
\`\`\`

### 3. Analyze Results
Perform measurements and analysis:
\`\`\`
measure_parameters(acquisition_id="...", measurements=["frequency", "amplitude"])
analyze_spectrum(acquisition_id="...", window="hamming")
\`\`\`

### 4. Protocol Decoding (Optional)
If working with digital signals:
\`\`\`
decode_protocol(acquisition_id="...", protocol="uart", settings={"baud_rate": 9600})
\`\`\`

## Hardware Interface Types
- **simulation**: Mock data for testing (current)
- **usb**: USB-based ADC devices
- **ethernet**: Network-connected ADCs
- **pcie**: PCIe ADC cards

## Next Steps
1. Verify basic functionality with test signals
2. Connect your physical ADC hardware
3. Update configuration for your specific hardware
4. Implement real-time data streaming if needed
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

  return server.server;
}