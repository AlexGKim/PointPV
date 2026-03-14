"""
pytest configuration: register custom marks to prevent PytestUnknownMarkWarning.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line(
        "markers", "flip: mark test as requiring FLIP/CAMB installation"
    )
