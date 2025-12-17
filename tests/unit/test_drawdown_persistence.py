"""
Unit tests for drawdown controller state persistence.
"""

import json
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.risk.drawdown import DrawdownController


class TestDrawdownPersistence:
    """Tests for drawdown state persistence."""

    @pytest.fixture
    def controller(self):
        """Create drawdown controller."""
        return DrawdownController(
            max_drawdown=0.15,
            reduce_at_drawdown=0.10,
            reduce_by_pct=0.50,
            close_all_at_drawdown=0.20,
        )

    def test_save_state(self, controller):
        """Test saving state to file."""
        # Set some state
        controller.update(1000000)
        controller.update(950000)  # 5% drawdown

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            controller.save_state(filepath)

            assert filepath.exists()

            with open(filepath, "r") as f:
                state = json.load(f)

            assert state["peak_value"] == 1000000
            assert state["current_drawdown"] == 0.05
            assert state["version"] == "1.0"
        finally:
            filepath.unlink()

    def test_load_state(self, controller):
        """Test loading state from file."""
        # Set initial state
        controller.update(1000000)
        controller.update(890000)  # 11% drawdown -> reduced

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            controller.save_state(filepath)

            # Create new controller and load
            new_controller = DrawdownController()
            success = new_controller.load_state(filepath)

            assert success
            assert new_controller._peak_value == controller._peak_value
            assert abs(new_controller._current_drawdown - controller._current_drawdown) < 0.001
            assert new_controller._is_reduced == controller._is_reduced
        finally:
            filepath.unlink()

    def test_load_nonexistent_file(self, controller):
        """Test loading from nonexistent file."""
        result = controller.load_state("/nonexistent/path/state.json")
        assert result is False

    def test_load_invalid_json(self, controller):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json")
            filepath = Path(f.name)

        try:
            result = controller.load_state(filepath)
            assert result is False
        finally:
            filepath.unlink()

    def test_from_state_file(self):
        """Test creating controller from state file."""
        original = DrawdownController(
            max_drawdown=0.20,
            reduce_at_drawdown=0.12,
        )
        original.update(1000000)
        original.update(870000)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            original.save_state(filepath)

            restored = DrawdownController.from_state_file(
                filepath,
                max_drawdown=0.20,
                reduce_at_drawdown=0.12,
            )

            assert restored._peak_value == original._peak_value
            assert restored._is_reduced == original._is_reduced
        finally:
            filepath.unlink()

    def test_get_state_dict(self, controller):
        """Test getting state as dictionary."""
        controller.update(1000000)
        controller.update(920000)

        state = controller.get_state_dict()

        assert "peak_value" in state
        assert "current_drawdown" in state
        assert "drawdown_pct" in state
        assert "is_reduced" in state
        assert "is_closed" in state

    def test_state_persistence_after_reduce(self, controller):
        """Test state persistence captures reduction status."""
        controller.update(1000000)
        controller.update(890000)  # Should trigger reduce

        assert controller._is_reduced

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            controller.save_state(filepath)

            new_controller = DrawdownController()
            new_controller.load_state(filepath)

            assert new_controller._is_reduced
        finally:
            filepath.unlink()

    def test_atomic_save(self, controller):
        """Test that save is atomic (uses temp file)."""
        controller.update(1000000)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "state.json"

            # First save
            controller.save_state(filepath)
            assert filepath.exists()

            # Update and save again
            controller.update(950000)
            controller.save_state(filepath)

            # Verify no .tmp file left
            tmp_file = filepath.with_suffix(".tmp")
            assert not tmp_file.exists()

    def test_config_mismatch_warning(self, controller):
        """Test warning when loaded state has different config."""
        # Save with one config
        controller.update(1000000)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            controller.save_state(filepath)

            # Load into controller with different config
            new_controller = DrawdownController(
                max_drawdown=0.25,  # Different from original 0.15
            )
            new_controller.load_state(filepath)

            # Should still load successfully (with warning logged)
            assert new_controller._peak_value == 1000000
        finally:
            filepath.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
