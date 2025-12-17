"""
Unit tests for compliance and audit trail system.
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.compliance.audit_trail import (
    AuditTrail,
    AuditEvent,
    AuditEventType,
)
from src.risk.pretrade_compliance import (
    PreTradeComplianceChecker,
    Order,
    ComplianceResult,
)


class TestAuditTrail:
    """Tests for audit trail system."""

    @pytest.fixture
    def temp_audit_dir(self):
        """Create temporary directory for audit logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_log_event(self, temp_audit_dir):
        """Test logging an event."""
        trail = AuditTrail(log_dir=temp_audit_dir)

        event = trail.log_event(
            event_type=AuditEventType.ORDER_SUBMITTED,
            source="TestModule",
            description="Test order",
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            price=150.0,
        )

        assert event.event_id is not None
        assert event.sequence_number == 1
        assert event.event_type == AuditEventType.ORDER_SUBMITTED
        assert event.symbol == "AAPL"

        trail.close()

    def test_hash_chain(self, temp_audit_dir):
        """Test that events form a valid hash chain."""
        trail = AuditTrail(log_dir=temp_audit_dir)

        # Log multiple events
        event1 = trail.log_event(
            event_type=AuditEventType.SYSTEM_STARTUP,
            source="System",
            description="System started",
        )

        event2 = trail.log_event(
            event_type=AuditEventType.ORDER_SUBMITTED,
            source="OrderManager",
            description="Order submitted",
            order_id="ORD-001",
        )

        # Check chain
        assert event2.previous_hash == event1.event_hash
        assert event1.previous_hash == "GENESIS"

        trail.close()

    def test_verify_chain(self, temp_audit_dir):
        """Test chain verification."""
        trail = AuditTrail(log_dir=temp_audit_dir)

        # Log events
        for i in range(10):
            trail.log_event(
                event_type=AuditEventType.ORDER_SUBMITTED,
                source="Test",
                description=f"Event {i}",
            )

        trail._sync()

        # Verify
        is_valid, message = trail.verify_chain()

        assert is_valid
        assert "10 events" in message

        trail.close()

    def test_query_events(self, temp_audit_dir):
        """Test querying events."""
        trail = AuditTrail(log_dir=temp_audit_dir)

        # Log various events
        trail.log_event(
            event_type=AuditEventType.ORDER_SUBMITTED,
            source="Test",
            description="Order 1",
            order_id="ORD-001",
            symbol="AAPL",
        )
        trail.log_event(
            event_type=AuditEventType.ORDER_FILLED,
            source="Test",
            description="Fill 1",
            order_id="ORD-001",
            symbol="AAPL",
        )
        trail.log_event(
            event_type=AuditEventType.ORDER_SUBMITTED,
            source="Test",
            description="Order 2",
            order_id="ORD-002",
            symbol="GOOGL",
        )

        trail._sync()

        # Query by symbol
        events = trail.query_events(symbol="AAPL")
        assert len(events) == 2

        # Query by event type
        events = trail.query_events(event_types=[AuditEventType.ORDER_SUBMITTED])
        assert len(events) == 2

        trail.close()

    def test_convenience_methods(self, temp_audit_dir):
        """Test convenience logging methods."""
        trail = AuditTrail(log_dir=temp_audit_dir)

        # Test order submitted
        event = trail.log_order_submitted(
            order_id="ORD-001",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
        )
        assert event.event_type == AuditEventType.ORDER_SUBMITTED

        # Test order filled
        event = trail.log_order_filled(
            order_id="ORD-001",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
        )
        assert event.event_type == AuditEventType.ORDER_FILLED

        # Test risk event
        event = trail.log_risk_event(
            event_type=AuditEventType.RISK_LIMIT_WARNING,
            description="VaR limit approaching",
        )
        assert event.event_type == AuditEventType.RISK_LIMIT_WARNING

        trail.close()


class TestPreTradeCompliance:
    """Tests for pre-trade compliance checker."""

    @pytest.fixture
    def checker(self):
        """Create compliance checker."""
        config = {
            "trade_limits": {
                "min_trade_size": 1000,
                "max_trade_size": 100000,
                "max_daily_trades": 100,
            },
            "exposure": {
                "max_single_position": 0.20,
                "max_gross_exposure": 1.5,
                "max_sector_exposure": 0.40,
            },
            "liquidity": {
                "max_adv_pct": 0.05,
                "min_adv": 100000,
            },
        }
        return PreTradeComplianceChecker(config=config, portfolio_value=1_000_000)

    def test_valid_order_passes(self, checker):
        """Test that valid order passes compliance."""
        checker.update_security_data("AAPL", adv=10_000_000, sector="Technology")

        order = Order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
        )

        report = checker.check_order(order)

        assert report.overall_result == ComplianceResult.PASSED

    def test_restricted_security_rejected(self, checker):
        """Test that restricted security is rejected."""
        checker.add_restricted_security("XYZ")

        order = Order(
            symbol="XYZ",
            side="BUY",
            quantity=100,
            price=50.0,
        )

        report = checker.check_order(order)

        assert report.overall_result == ComplianceResult.REJECTED
        assert "restricted" in report.checks[0].message.lower()

    def test_order_too_small_rejected(self, checker):
        """Test that orders below minimum are rejected."""
        checker.update_security_data("AAPL", adv=10_000_000)

        order = Order(
            symbol="AAPL",
            side="BUY",
            quantity=5,
            price=100.0,  # $500 order
        )

        report = checker.check_order(order)

        assert report.overall_result == ComplianceResult.REJECTED

    def test_order_too_large_modified(self, checker):
        """Test that orders exceeding max are modified."""
        checker.update_security_data("AAPL", adv=10_000_000)

        order = Order(
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=150.0,  # $150,000 order
        )

        report = checker.check_order(order)

        assert report.overall_result == ComplianceResult.MODIFIED
        assert report.modified_quantity < order.quantity

    def test_position_concentration_modified(self, checker):
        """Test that concentration breaches are modified."""
        checker.update_security_data("AAPL", adv=10_000_000)
        checker.update_position("AAPL", quantity=1000, market_value=150_000)

        order = Order(
            symbol="AAPL",
            side="BUY",
            quantity=500,
            price=150.0,  # Would bring position to 22.5%
        )

        report = checker.check_order(order)

        # Should be modified to stay under 20%
        assert report.overall_result in [ComplianceResult.MODIFIED, ComplianceResult.PASSED]

    def test_daily_trade_count_limit(self, checker):
        """Test daily trade count limit."""
        from datetime import datetime
        checker._daily_trade_count = 100  # At limit
        checker._last_reset_date = datetime.now()  # Prevent reset
        checker.update_security_data("AAPL", adv=10_000_000)

        order = Order(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            price=150.0,
        )

        report = checker.check_order(order)

        assert any(
            c.check_name == "daily_trade_count" and c.result == ComplianceResult.REJECTED
            for c in report.checks
        )

    def test_liquidity_check_adv(self, checker):
        """Test ADV liquidity check."""
        checker.update_security_data("ILLIQ", adv=50_000)  # Below minimum

        order = Order(
            symbol="ILLIQ",
            side="BUY",
            quantity=100,
            price=50.0,
        )

        report = checker.check_order(order)

        assert report.overall_result == ComplianceResult.REJECTED

    def test_executed_trade_tracking(self, checker):
        """Test that executed trades are tracked."""
        order = Order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
        )

        initial_count = checker._daily_trade_count
        checker.record_executed_trade(order)

        assert checker._daily_trade_count == initial_count + 1
        assert checker._daily_turnover == 15000


class TestOrder:
    """Tests for Order dataclass."""

    def test_order_creation(self):
        """Test order creation."""
        order = Order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=150.0,
            order_type="LIMIT",
        )

        assert order.symbol == "AAPL"
        assert order.side == "BUY"
        assert order.quantity == 100
        assert order.price == 150.0
        assert order.order_type == "LIMIT"

    def test_market_order_no_price(self):
        """Test market order without price."""
        order = Order(
            symbol="AAPL",
            side="BUY",
            quantity=100,
        )

        assert order.price is None
        assert order.order_type == "MARKET"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
