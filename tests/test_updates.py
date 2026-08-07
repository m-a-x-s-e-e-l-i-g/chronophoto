from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from chronophoto import __version__
from chronophoto.updates import evaluate_release, version_key


def test_runtime_and_packaging_versions_match() -> None:
    project_root = Path(__file__).parents[1]
    with (project_root / "pyproject.toml").open("rb") as project_file:
        project_version = tomllib.load(project_file)["project"]["version"]
    installer = (project_root / "packaging/windows/chronophoto.iss").read_text(encoding="utf-8")

    assert __version__ == project_version
    assert f'#define MyAppVersion "{project_version}"' in installer


def test_version_key_accepts_release_tags() -> None:
    assert version_key("v1.2.3") == (1, 2, 3)
    assert version_key("1.2") == (1, 2, 0)


def test_release_comparison_detects_updates_and_current_builds() -> None:
    available = evaluate_release("0.1.0", "v0.2.0", "https://example.com/v0.2.0")
    current = evaluate_release("0.2.0", "v0.2.0", "https://example.com/v0.2.0")
    development = evaluate_release("0.3.0", "v0.2.0", "https://example.com/v0.2.0")

    assert available.update_available
    assert available.latest_version == "0.2.0"
    assert not current.update_available
    assert not development.update_available


def test_version_key_rejects_unusable_tags() -> None:
    with pytest.raises(ValueError, match="Unsupported version"):
        version_key("latest")
