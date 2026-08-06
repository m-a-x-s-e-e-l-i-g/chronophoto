from __future__ import annotations

import re
from dataclasses import dataclass

GITHUB_REPOSITORY_URL = "https://github.com/m-a-x-s-e-e-l-i-g/chronophoto"
LATEST_RELEASE_API_URL = (
    "https://api.github.com/repos/m-a-x-s-e-e-l-i-g/chronophoto/releases/latest"
)

_VERSION_PATTERN = re.compile(
    r"^[vV]?(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?(?:[-+].*)?$"
)


@dataclass(slots=True, frozen=True)
class UpdateResult:
    latest_version: str
    release_url: str
    update_available: bool


def version_key(value: str) -> tuple[int, int, int]:
    """Return the comparable numeric portion of a release version or tag."""

    match = _VERSION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Unsupported version: {value}")
    return tuple(int(match.group(name) or 0) for name in ("major", "minor", "patch"))


def evaluate_release(
    current_version: str,
    latest_tag: str,
    release_url: str,
) -> UpdateResult:
    """Compare the installed version with a GitHub release tag."""

    latest = latest_tag.strip().removeprefix("v").removeprefix("V")
    return UpdateResult(
        latest,
        release_url,
        version_key(latest) > version_key(current_version),
    )
