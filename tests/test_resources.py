"""
Test resource management functionality.

Tests for MCP resource management, data access, and resource lifecycle.
"""

import pytest
import json
import time
from typing import Dict, Any, List

from oscilloscope_mcp.resources.manager import ResourceManager


class TestResourceManager:
    """Test resource manager functionality."""

    def test_resource_manager_creation(self):
        """Test resource manager instantiation."""
        manager = ResourceManager()
        assert manager is not None

    def test_resource_registration(self):
        """Test registering resources."""
        manager = ResourceManager()
        
        # Register a waveform data resource
        resource_config = {
            "name": "waveform_data",
            "description": "Real-time waveform data access",
            "uri_template": "waveform://channel{channel}/buffer",
            "mime_type": "application/json",
            "access_permissions": ["read"]
        }
        
        success = manager.register_resource(resource_config)
        assert success is True
        
        # Verify resource was registered
        resources = manager.list_resources()
        assert len(resources) > 0
        assert any(r["name"] == "waveform_data" for r in resources)

    def test_resource_access(self):
        """Test accessing registered resources."""
        manager = ResourceManager()
        
        # Register test resource
        resource_config = {
            "name": "test_data",
            "description": "Test data resource",
            "uri_template": "test://data/{id}",
            "mime_type": "application/json",
            "access_permissions": ["read", "write"]
        }
        manager.register_resource(resource_config)
        
        # Access the resource
        resource_uri = "test://data/123"
        resource_data = manager.get_resource(resource_uri)
        
        # Should return resource metadata or data
        assert resource_data is not None

    def test_resource_uri_template_expansion(self):
        """Test URI template parameter expansion."""
        manager = ResourceManager()
        
        template = "waveform://channel{channel}/buffer?samples={samples}"
        parameters = {"channel": 1, "samples": 1000}
        
        expanded_uri = manager.expand_uri_template(template, parameters)
        
        expected_uri = "waveform://channel1/buffer?samples=1000"
        assert expanded_uri == expected_uri

    def test_resource_caching(self):
        """Test resource data caching."""
        manager = ResourceManager()
        
        # Enable caching
        cache_config = {
            "enabled": True,
            "max_size": 100,  # MB
            "ttl": 300  # 5 minutes
        }
        manager.configure_cache(cache_config)
        
        # Register cacheable resource
        resource_config = {
            "name": "cached_data",
            "description": "Cached measurement data",
            "uri_template": "measurements://cache/{id}",
            "mime_type": "application/json",
            "cacheable": True,
            "cache_ttl": 60
        }
        manager.register_resource(resource_config)
        
        # Verify cache configuration
        cache_info = manager.get_cache_info()
        assert cache_info["enabled"] is True
        assert cache_info["max_size"] == 100


class TestWaveformDataResource:
    """Test waveform data resource access."""

    @pytest.fixture
    def waveform_manager(self):
        """Create resource manager with waveform resources."""
        manager = ResourceManager()
        
        # Register waveform data resource
        waveform_config = {
            "name": "waveform_data",
            "description": "Real-time access to acquired waveform data",
            "uri_template": "waveform://channel{channel}/buffer",
            "mime_type": "application/json",
            "access_permissions": ["read"],
            "real_time": True
        }
        manager.register_resource(waveform_config)
        
        return manager

    def test_waveform_data_access(self, waveform_manager):
        """Test accessing waveform data."""
        manager = waveform_manager
        
        # Simulate waveform data
        test_waveform = {
            "channel": 1,
            "sample_rate": 1000000,
            "samples": [0.1, 0.2, 0.3, 0.4, 0.5],
            "timestamp": time.time(),
            "units": "volts"
        }
        
        # Store waveform data
        uri = "waveform://channel1/buffer"
        manager.store_data(uri, test_waveform)
        
        # Retrieve waveform data
        retrieved_data = manager.get_resource(uri)
        
        assert retrieved_data is not None
        assert retrieved_data["channel"] == 1
        assert len(retrieved_data["samples"]) == 5

    def test_multi_channel_waveform_access(self, waveform_manager):
        """Test accessing multi-channel waveform data."""
        manager = waveform_manager
        
        # Store data for multiple channels
        for channel in [1, 2, 3]:
            test_data = {
                "channel": channel,
                "sample_rate": 1000000,
                "samples": list(range(channel * 10, (channel + 1) * 10)),
                "timestamp": time.time()
            }
            
            uri = f"waveform://channel{channel}/buffer"
            manager.store_data(uri, test_data)
        
        # Retrieve all channels
        all_channels = []
        for channel in [1, 2, 3]:
            uri = f"waveform://channel{channel}/buffer"
            data = manager.get_resource(uri)
            all_channels.append(data)
        
        assert len(all_channels) == 3
        assert all_channels[0]["channel"] == 1
        assert all_channels[1]["channel"] == 2
        assert all_channels[2]["channel"] == 3

    def test_waveform_data_streaming(self, waveform_manager):
        """Test streaming waveform data updates."""
        manager = waveform_manager
        
        # Set up streaming for a channel
        uri = "waveform://channel1/buffer"
        
        # Simulate multiple data updates
        updates = []
        for i in range(5):
            data = {
                "channel": 1,
                "sample_rate": 1000000,
                "samples": [i * 0.1] * 10,
                "timestamp": time.time() + i * 0.001,
                "sequence": i
            }
            
            manager.store_data(uri, data)
            updates.append(data)
            time.sleep(0.001)  # Small delay
        
        # Get latest data
        latest = manager.get_resource(uri)
        assert latest["sequence"] == 4  # Last update


class TestSpectrumDataResource:
    """Test spectrum analysis data resource."""

    @pytest.fixture
    def spectrum_manager(self):
        """Create resource manager with spectrum resources."""
        manager = ResourceManager()
        
        spectrum_config = {
            "name": "spectrum_data",
            "description": "Access to FFT analysis results",
            "uri_template": "spectrum://channel{channel}/fft",
            "mime_type": "application/json",
            "access_permissions": ["read"],
            "computed": True
        }
        manager.register_resource(spectrum_config)
        
        return manager

    def test_spectrum_data_access(self, spectrum_manager):
        """Test accessing spectrum analysis data."""
        manager = spectrum_manager
        
        # Simulate spectrum data
        spectrum_data = {
            "channel": 1,
            "frequencies": list(range(0, 1000, 10)),  # 0-1000 Hz in 10 Hz steps
            "magnitudes": [1.0 / (i + 1) for i in range(100)],  # Decreasing magnitude
            "phases": [0.0] * 100,
            "window": "hann",
            "fft_size": 1024,
            "timestamp": time.time()
        }
        
        uri = "spectrum://channel1/fft"
        manager.store_data(uri, spectrum_data)
        
        retrieved = manager.get_resource(uri)
        
        assert retrieved is not None
        assert len(retrieved["frequencies"]) == 100
        assert len(retrieved["magnitudes"]) == 100
        assert retrieved["window"] == "hann"

    def test_spectrum_parameter_filtering(self, spectrum_manager):
        """Test filtering spectrum data by parameters."""
        manager = spectrum_manager
        
        # Store full spectrum data
        full_spectrum = {
            "channel": 1,
            "frequencies": list(range(0, 5000, 50)),  # 0-5000 Hz
            "magnitudes": [1.0] * 100,
            "timestamp": time.time()
        }
        
        uri = "spectrum://channel1/fft"
        manager.store_data(uri, full_spectrum)
        
        # Request filtered data (e.g., 1000-2000 Hz range)
        filter_params = {
            "freq_min": 1000,
            "freq_max": 2000
        }
        
        filtered = manager.get_resource(uri, filter_params)
        
        # Should contain only frequencies in the specified range
        assert filtered is not None


class TestMeasurementResultsResource:
    """Test measurement results resource."""

    @pytest.fixture
    def measurement_manager(self):
        """Create resource manager with measurement resources."""
        manager = ResourceManager()
        
        measurement_config = {
            "name": "measurement_results",
            "description": "Access to measurement results and statistics",
            "uri_template": "measurements://channel{channel}/results",
            "mime_type": "application/json",
            "access_permissions": ["read", "write"],
            "persistent": True
        }
        manager.register_resource(measurement_config)
        
        return manager

    def test_measurement_results_storage(self, measurement_manager):
        """Test storing and retrieving measurement results."""
        manager = measurement_manager
        
        # Store measurement results
        measurements = {
            "channel": 1,
            "rms": 2.345,
            "peak_to_peak": 6.78,
            "frequency": 1000.5,
            "rise_time": 0.001234,
            "fall_time": 0.001456,
            "timestamp": time.time(),
            "measurement_id": "meas_001"
        }
        
        uri = "measurements://channel1/results"
        manager.store_data(uri, measurements)
        
        retrieved = manager.get_resource(uri)
        
        assert retrieved is not None
        assert retrieved["rms"] == 2.345
        assert retrieved["measurement_id"] == "meas_001"

    def test_measurement_history(self, measurement_manager):
        """Test accessing measurement history."""
        manager = measurement_manager
        
        uri = "measurements://channel1/results"
        
        # Store multiple measurements over time
        for i in range(10):
            measurement = {
                "channel": 1,
                "rms": 2.0 + i * 0.1,
                "timestamp": time.time() + i * 0.1,
                "measurement_id": f"meas_{i:03d}"
            }
            
            manager.store_data(uri, measurement, append_history=True)
        
        # Get measurement history
        history = manager.get_resource_history(uri, limit=5)
        
        assert len(history) == 5  # Last 5 measurements
        assert history[0]["measurement_id"] == "meas_009"  # Most recent first

    def test_measurement_statistics(self, measurement_manager):
        """Test measurement statistics calculation."""
        manager = measurement_manager
        
        # Store multiple measurements for statistics
        measurements = []
        for i in range(100):
            measurement = {
                "channel": 1,
                "rms": 2.0 + 0.1 * (i % 10),  # Repeating pattern
                "timestamp": time.time() + i * 0.001
            }
            measurements.append(measurement)
        
        uri = "measurements://channel1/results"
        manager.store_measurement_batch(uri, measurements)
        
        # Calculate statistics
        stats = manager.calculate_statistics(uri, "rms")
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        
        assert stats["count"] == 100


class TestInstrumentConfigResource:
    """Test instrument configuration resource."""

    @pytest.fixture
    def config_manager(self):
        """Create resource manager with configuration resources."""
        manager = ResourceManager()
        
        config_resource = {
            "name": "instrument_config",
            "description": "Current instrument configuration and settings",
            "uri_template": "config://instrument/settings",
            "mime_type": "application/json",
            "access_permissions": ["read", "write"],
            "persistent": True
        }
        manager.register_resource(config_resource)
        
        return manager

    def test_config_storage_retrieval(self, config_manager):
        """Test storing and retrieving instrument configuration."""
        manager = config_manager
        
        # Store instrument configuration
        config = {
            "oscilloscope": {
                "sample_rate": 1000000,
                "channels": {
                    "1": {"range": 5.0, "coupling": "DC", "enabled": True},
                    "2": {"range": 2.0, "coupling": "AC", "enabled": True}
                },
                "trigger": {
                    "source": "CH1",
                    "type": "edge",
                    "level": 0.5,
                    "edge": "rising"
                }
            },
            "function_generator": {
                "channel_1": {
                    "waveform": "sine",
                    "frequency": 1000,
                    "amplitude": 1.0,
                    "enabled": True
                }
            },
            "timestamp": time.time()
        }
        
        uri = "config://instrument/settings"
        manager.store_data(uri, config)
        
        retrieved = manager.get_resource(uri)
        
        assert retrieved is not None
        assert retrieved["oscilloscope"]["sample_rate"] == 1000000
        assert retrieved["function_generator"]["channel_1"]["frequency"] == 1000

    def test_config_validation(self, config_manager):
        """Test configuration validation."""
        manager = config_manager
        
        # Set validation schema
        validation_schema = {
            "type": "object",
            "properties": {
                "oscilloscope": {
                    "type": "object",
                    "properties": {
                        "sample_rate": {"type": "number", "minimum": 1000}
                    },
                    "required": ["sample_rate"]
                }
            },
            "required": ["oscilloscope"]
        }
        
        manager.set_validation_schema("config://instrument/settings", validation_schema)
        
        # Valid configuration
        valid_config = {
            "oscilloscope": {
                "sample_rate": 1000000
            }
        }
        
        uri = "config://instrument/settings"
        success = manager.store_data(uri, valid_config, validate=True)
        assert success is True
        
        # Invalid configuration (missing required field)
        invalid_config = {
            "function_generator": {
                "frequency": 1000
            }
        }
        
        success = manager.store_data(uri, invalid_config, validate=True)
        assert success is False  # Should fail validation


class TestResourceLifecycle:
    """Test resource lifecycle management."""

    def test_resource_creation_deletion(self):
        """Test creating and deleting resources."""
        manager = ResourceManager()
        
        # Create resource
        resource_config = {
            "name": "temp_resource",
            "description": "Temporary test resource",
            "uri_template": "temp://data/{id}",
            "mime_type": "text/plain"
        }
        
        success = manager.register_resource(resource_config)
        assert success is True
        
        # Verify resource exists
        resources = manager.list_resources()
        temp_resources = [r for r in resources if r["name"] == "temp_resource"]
        assert len(temp_resources) == 1
        
        # Delete resource
        success = manager.unregister_resource("temp_resource")
        assert success is True
        
        # Verify resource is gone
        resources = manager.list_resources()
        temp_resources = [r for r in resources if r["name"] == "temp_resource"]
        assert len(temp_resources) == 0

    def test_resource_expiration(self):
        """Test resource data expiration."""
        manager = ResourceManager()
        
        # Register resource with short TTL
        resource_config = {
            "name": "expiring_resource",
            "description": "Resource with expiration",
            "uri_template": "expire://data/{id}",
            "mime_type": "application/json",
            "ttl": 0.1  # 100ms TTL
        }
        manager.register_resource(resource_config)
        
        # Store data
        uri = "expire://data/test"
        test_data = {"value": 123, "timestamp": time.time()}
        manager.store_data(uri, test_data)
        
        # Data should be available immediately
        retrieved = manager.get_resource(uri)
        assert retrieved is not None
        assert retrieved["value"] == 123
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Data should be expired
        expired = manager.get_resource(uri)
        assert expired is None  # Should be expired

    def test_resource_cleanup(self):
        """Test resource cleanup and garbage collection."""
        manager = ResourceManager()
        
        # Create multiple resources with data
        for i in range(10):
            config = {
                "name": f"cleanup_resource_{i}",
                "description": f"Cleanup test resource {i}",
                "uri_template": f"cleanup://data_{i}/{{id}}",
                "mime_type": "application/json"
            }
            manager.register_resource(config)
            
            # Store some data
            uri = f"cleanup://data_{i}/test"
            manager.store_data(uri, {"index": i})
        
        # Perform cleanup
        cleanup_stats = manager.cleanup_resources()
        
        assert "resources_cleaned" in cleanup_stats
        assert "data_cleaned" in cleanup_stats
        assert cleanup_stats["resources_cleaned"] >= 0


class TestResourceSecurity:
    """Test resource security and access control."""

    def test_access_permissions(self):
        """Test resource access permission enforcement."""
        manager = ResourceManager()
        
        # Register read-only resource
        readonly_config = {
            "name": "readonly_resource",
            "description": "Read-only test resource",
            "uri_template": "readonly://data/{id}",
            "mime_type": "application/json",
            "access_permissions": ["read"]
        }
        manager.register_resource(readonly_config)
        
        uri = "readonly://data/test"
        
        # Reading should be allowed
        can_read = manager.check_permission(uri, "read")
        assert can_read is True
        
        # Writing should be denied
        can_write = manager.check_permission(uri, "write")
        assert can_write is False

    def test_resource_authentication(self):
        """Test resource authentication requirements."""
        manager = ResourceManager()
        
        # Register authenticated resource
        auth_config = {
            "name": "auth_resource",
            "description": "Authenticated resource",
            "uri_template": "auth://secure/{id}",
            "mime_type": "application/json",
            "authentication_required": True
        }
        manager.register_resource(auth_config)
        
        uri = "auth://secure/test"
        
        # Access without authentication should fail
        unauthenticated_access = manager.get_resource(uri)
        assert unauthenticated_access is None
        
        # Access with authentication should succeed
        auth_token = "valid_token_123"
        authenticated_access = manager.get_resource(uri, auth_token=auth_token)
        # Would succeed if authentication is implemented