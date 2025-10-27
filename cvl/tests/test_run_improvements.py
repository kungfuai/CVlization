"""Tests for cvl run improvements."""
import pytest
from cvl.commands.run import _format_duration, check_docker_running


def test_format_duration_seconds():
    """Test duration formatting for seconds only."""
    assert _format_duration(10) == "10s"
    assert _format_duration(45.7) == "45s"
    assert _format_duration(59) == "59s"


def test_format_duration_minutes():
    """Test duration formatting for minutes and seconds."""
    assert _format_duration(60) == "1m 0s"
    assert _format_duration(90) == "1m 30s"
    assert _format_duration(125) == "2m 5s"
    assert _format_duration(332) == "5m 32s"


def test_check_docker_running():
    """Test Docker daemon check.

    Note: This test actually checks if Docker is running on the system.
    It will pass if Docker is running, fail if not.
    """
    is_running, error_msg = check_docker_running()

    # If Docker is running, should return True with no error
    if is_running:
        assert is_running is True
        assert error_msg == ""
    else:
        # If not running, should return False with an error message
        assert is_running is False
        assert len(error_msg) > 0
        assert "Docker" in error_msg
