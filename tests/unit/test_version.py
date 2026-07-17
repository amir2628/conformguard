import re
from pathlib import Path

import conformguard

_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_dunder_version_matches_pyproject_declaration():
    text = _PYPROJECT_PATH.read_text()
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match is not None, f"no top-level version = \"...\" line found in {_PYPROJECT_PATH}"
    assert conformguard.__version__ == match.group(1)
