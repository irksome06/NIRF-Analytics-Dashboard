import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

from engine.data_loader import load_master


@pytest.fixture(scope="session")
def df():
    return load_master()
