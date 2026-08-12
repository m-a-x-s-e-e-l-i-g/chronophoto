from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chronophoto.processing import ComposeSettings
from chronophoto.processing.sources import VideoInfo


@dataclass(slots=True)
class SourceState:
    kind: str
    paths: list[Path]
    video_info: VideoInfo | None = None


@dataclass(slots=True, frozen=True)
class RenderRequest:
    kind: str
    paths: tuple[Path, ...]
    start: float
    end: float
    pose_count: int | None
    settings: ComposeSettings
    alignment: str
    max_dimension: int | None
    cache_key: tuple[object, ...]
    video_selection_key: tuple[object, ...] | None
    video_cache_key: tuple[object, ...] | None
    video_duration: float
    video_frame_rate: float
    enabled_video_indices: tuple[int, ...] | None
    focus_pose_index: int | None
    source_dimensions: tuple[int, int]
    pixel_aspect_ratio: tuple[int, int]


class SourceController:
    """Own the active source identity independently of the Qt view."""

    def __init__(self) -> None:
        self.state: SourceState | None = None

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self.state.paths) if self.state is not None else ()

    @property
    def is_video(self) -> bool:
        return self.state is not None and self.state.kind == "video"

    def replace(self, state: SourceState | None) -> None:
        self.state = state
