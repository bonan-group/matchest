"""
Test configuration and shared fixtures for VASP checker tests.

This module provides common configuration and fixtures that are shared
across multiple test modules.
"""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def test_fixtures_dir():
    """Path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_calculations(test_fixtures_dir):
    """Dictionary of sample calculation directories for testing."""
    return {
        "basic": test_fixtures_dir / "basic_calculation",
        "large": test_fixtures_dir / "large_calculation",
        "small": test_fixtures_dir / "small_calculation",
        "issues": test_fixtures_dir / "parallelization_issues",
        "multi": test_fixtures_dir / "multi_calc_directory",
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture
def basic_calc_dir(test_fixtures_dir):
    return test_fixtures_dir / "basic_calculation"


@pytest.fixture
def large_calc_dir(test_fixtures_dir):
    return test_fixtures_dir / "large_calculation"


@pytest.fixture
def small_calc_dir(test_fixtures_dir):
    return test_fixtures_dir / "small_calculation"


@pytest.fixture
def multi_calc_dir(test_fixtures_dir):
    return test_fixtures_dir / "multi_calc_directory"
