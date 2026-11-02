"""Fixtures of the tool tests."""
import pytest


@pytest.fixture
def forge_python(monkeypatch, tmp_path):
    """A `forge` interpreter declared for the cell (`NBX_ATELIER_PYTHON_FORGE`, tools/atelier.python): the cells that
    mock Forge's subprocess still resolve its interpreter to build the command, and a machine that has no Forge
    environment by design (the Mac: environments.md) declares none — without this the cell is refused by the machine,
    not by the code it tests."""
    path = tmp_path / "forge-venv" / "bin" / "python"
    monkeypatch.setenv("NBX_ATELIER_PYTHON_FORGE", str(path))
    return path
