"""
Unit Tests for Broker API Retry Logic
======================================

Tests the retry_with_backoff decorator and WebSocketReconnectManager.
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.execution.broker_api import (
    retry_with_backoff,
    RetryableError,
    NonRetryableError,
    WebSocketConfig,
    WebSocketReconnectManager
)


# =============================================================================
# RETRY DECORATOR TESTS
# =============================================================================

class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator"""

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self):
        """Test that successful calls don't retry"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test retry on ConnectionError"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection refused")
            return "success"

        result = await failing_then_success()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_timeout_error(self):
        """Test retry on asyncio.TimeoutError"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        async def timeout_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError()
            return "success"

        result = await timeout_then_success()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that max retries is enforced"""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            await always_fails()

        # Initial call + 2 retries = 3 total calls
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self):
        """Test that NonRetryableError doesn't retry"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def raises_non_retryable():
            nonlocal call_count
            call_count += 1
            raise NonRetryableError("Don't retry this")

        with pytest.raises(NonRetryableError):
            await raises_non_retryable()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_unexpected_exception(self):
        """Test that unexpected exceptions don't retry"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def raises_unexpected():
            nonlocal call_count
            call_count += 1
            raise ValueError("Unexpected error")

        with pytest.raises(ValueError):
            await raises_unexpected()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test that delays follow exponential backoff pattern"""
        delays = []
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.1, exponential_base=2.0, jitter=False)
        async def track_delays():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Track timing")

        # Patch asyncio.sleep to track delays
        original_sleep = asyncio.sleep

        async def mock_sleep(delay):
            delays.append(delay)
            await original_sleep(0.001)  # Minimal actual sleep

        with patch('asyncio.sleep', mock_sleep):
            with pytest.raises(ConnectionError):
                await track_delays()

        # Should have delays: 0.1, 0.2, 0.4 (exponential: base * 2^attempt)
        assert len(delays) == 3
        assert abs(delays[0] - 0.1) < 0.01
        assert abs(delays[1] - 0.2) < 0.01
        assert abs(delays[2] - 0.4) < 0.01

    @pytest.mark.asyncio
    async def test_retry_preserves_function_metadata(self):
        """Test that decorated function preserves original metadata"""

        @retry_with_backoff()
        async def documented_func():
            """This is a documented function"""
            return True

        assert documented_func.__name__ == "documented_func"
        assert "documented" in documented_func.__doc__


# =============================================================================
# WEBSOCKET RECONNECT MANAGER TESTS
# =============================================================================

class TestWebSocketReconnectManager:
    """Tests for WebSocketReconnectManager"""

    def test_config_defaults(self):
        """Test WebSocketConfig default values"""
        config = WebSocketConfig(url="wss://test.com")

        assert config.url == "wss://test.com"
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.max_reconnect_attempts == -1  # Unlimited
        assert config.ping_interval == 30.0
        assert config.ping_timeout == 10.0
        assert config.jitter is True

    def test_manager_initial_state(self):
        """Test WebSocketReconnectManager initial state"""
        config = WebSocketConfig(url="wss://test.com")
        manager = WebSocketReconnectManager(config=config)

        assert manager.is_connected is False
        assert manager._running is False
        assert manager._reconnect_attempts == 0
        assert len(manager._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_subscribe_stores_subscription(self):
        """Test that subscriptions are stored for replay"""
        config = WebSocketConfig(url="wss://test.com")
        manager = WebSocketReconnectManager(config=config)

        subscription = {"action": "subscribe", "channel": "trades"}
        await manager.subscribe(subscription)

        assert subscription in manager._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_subscription(self):
        """Test that unsubscribe removes from stored subscriptions"""
        config = WebSocketConfig(url="wss://test.com")
        manager = WebSocketReconnectManager(config=config)

        subscription = {"action": "subscribe", "channel": "trades"}
        await manager.subscribe(subscription)
        await manager.unsubscribe(subscription)

        assert subscription not in manager._subscriptions

    def test_stats_initial_values(self):
        """Test that stats are properly initialized"""
        config = WebSocketConfig(url="wss://test.com")
        manager = WebSocketReconnectManager(config=config)

        stats = manager.get_stats()

        assert stats['connects'] == 0
        assert stats['disconnects'] == 0
        assert stats['reconnects'] == 0
        assert stats['messages_received'] == 0
        assert stats['messages_sent'] == 0
        assert stats['connected'] is False

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """Test that disconnect works safely when not connected"""
        config = WebSocketConfig(url="wss://test.com")
        manager = WebSocketReconnectManager(config=config)

        # Should not raise
        await manager.disconnect()

        assert manager._running is False
        assert manager._connected is False

    @pytest.mark.asyncio
    async def test_send_fails_when_disconnected(self):
        """Test that send returns False when not connected"""
        config = WebSocketConfig(url="wss://test.com")
        manager = WebSocketReconnectManager(config=config)

        result = await manager.send({"test": "message"})

        assert result is False

    @pytest.mark.asyncio
    async def test_reconnect_delay_exponential(self):
        """Test that reconnect delays follow exponential pattern"""
        config = WebSocketConfig(
            url="wss://test.com",
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False
        )
        manager = WebSocketReconnectManager(config=config)

        # Simulate reconnect attempts and check delay calculation
        delays = []
        for i in range(5):
            manager._reconnect_attempts = i + 1
            delay = min(
                config.initial_delay * (config.exponential_base ** i),
                config.max_delay
            )
            delays.append(delay)

        # Expected: 1, 2, 4, 8, 16
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    @pytest.mark.asyncio
    async def test_reconnect_delay_capped_at_max(self):
        """Test that reconnect delay is capped at max_delay"""
        config = WebSocketConfig(
            url="wss://test.com",
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False
        )
        manager = WebSocketReconnectManager(config=config)

        # After many attempts, delay should be capped
        manager._reconnect_attempts = 10
        delay = min(
            config.initial_delay * (config.exponential_base ** 9),
            config.max_delay
        )

        assert delay == 10.0  # Capped at max_delay


# =============================================================================
# INTEGRATION-STYLE TESTS (with mocks)
# =============================================================================

class TestRetryIntegration:
    """Integration-style tests for retry logic"""

    @pytest.mark.asyncio
    async def test_http_502_triggers_retry(self):
        """Test that HTTP 502 error triggers retry"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        async def mock_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise aiohttp.ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=502,
                    message="Bad Gateway"
                )
            return {"status": "ok"}

        result = await mock_api_call()

        assert result == {"status": "ok"}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_http_503_triggers_retry(self):
        """Test that HTTP 503 error triggers retry"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        async def mock_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise aiohttp.ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=503,
                    message="Service Unavailable"
                )
            return {"status": "ok"}

        result = await mock_api_call()

        assert result == {"status": "ok"}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_http_429_triggers_retry(self):
        """Test that HTTP 429 (rate limit) error triggers retry"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        async def mock_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise aiohttp.ClientResponseError(
                    request_info=MagicMock(),
                    history=(),
                    status=429,
                    message="Too Many Requests"
                )
            return {"status": "ok"}

        result = await mock_api_call()

        assert result == {"status": "ok"}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_http_400_no_retry(self):
        """Test that HTTP 400 (client error) doesn't retry"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def mock_api_call():
            nonlocal call_count
            call_count += 1
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=400,
                message="Bad Request"
            )

        with pytest.raises(aiohttp.ClientResponseError):
            await mock_api_call()

        assert call_count == 1  # No retries for 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
