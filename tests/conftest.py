"""Shared pytest fixtures."""

from pathlib import Path

import pytest

from financial_agent.data_loader import DataLoader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def loader() -> DataLoader:
    return DataLoader(DATA_DIR)
