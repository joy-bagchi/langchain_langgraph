import pytest

from market_physics_core import __version__
from market_physics_core.cli import main


def test_package_import_and_cli_metadata(capsys):
    assert __version__ == "0.1.0"
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "Deterministic analysis engine" in capsys.readouterr().out
