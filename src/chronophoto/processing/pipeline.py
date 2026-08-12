from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

ExecutionMode = Literal["materialized", "streaming"]
ProgressCallback = Callable[[int, str], None]
StageRunner = Callable[[Mapping[str, "StageArtifact"], ProgressCallback | None], "StageArtifact"]


@dataclass(slots=True, frozen=True)
class StageArtifact:
    kind: str
    value: Any
    cache_key: tuple[object, ...]
    estimated_bytes: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RenderStage:
    name: str
    output_kind: str
    run: StageRunner
    dependencies: tuple[str, ...] = ()
    mode: ExecutionMode = "materialized"
    cache_key: tuple[object, ...] = ()


@dataclass(slots=True, frozen=True)
class StageTiming:
    name: str
    seconds: float
    reused: bool
    mode: ExecutionMode


@dataclass(slots=True, frozen=True)
class RenderExecution:
    artifacts: Mapping[str, StageArtifact]
    timings: tuple[StageTiming, ...]


class RenderPlan:
    """Small typed DAG shared by preview and export orchestration."""

    def __init__(self, stages: tuple[RenderStage, ...]) -> None:
        if not stages:
            raise ValueError("render plan requires at least one stage")
        self.stages = stages
        self._validate()

    def _validate(self) -> None:
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("render stage names must be unique")
        available: set[str] = set()
        for stage in self.stages:
            missing = set(stage.dependencies) - available
            if missing:
                raise ValueError(
                    f"stage {stage.name} depends on unavailable stages: {sorted(missing)}"
                )
            available.add(stage.name)

    def execute(
        self,
        *,
        progress: ProgressCallback | None = None,
        cache: dict[tuple[object, ...], StageArtifact] | None = None,
    ) -> RenderExecution:
        artifacts: dict[str, StageArtifact] = {}
        timings: list[StageTiming] = []
        total = len(self.stages)
        for index, stage in enumerate(self.stages):
            started = perf_counter()
            reused = False
            artifact = cache.get(stage.cache_key) if cache is not None and stage.cache_key else None
            if artifact is None:
                dependency_artifacts = {name: artifacts[name] for name in stage.dependencies}

                stage_index = index
                stage_name = stage.name

                def stage_progress(
                    value: int,
                    message: str,
                    *,
                    _stage_index: int = stage_index,
                    _stage_name: str = stage_name,
                ) -> None:
                    if progress is None:
                        return
                    overall = round(((_stage_index + max(0, min(100, value)) / 100) / total) * 100)
                    progress(overall, f"{_stage_name} · {message}")

                artifact = stage.run(dependency_artifacts, stage_progress if progress else None)
                if artifact.kind != stage.output_kind:
                    raise TypeError(
                        f"stage {stage.name} produced {artifact.kind}, expected {stage.output_kind}"
                    )
                if cache is not None and stage.cache_key:
                    cache[stage.cache_key] = artifact
            else:
                reused = True
                if progress is not None:
                    progress(round(((index + 1) / total) * 100), f"{stage.name} · reused cache")
            artifacts[stage.name] = artifact
            timings.append(StageTiming(stage.name, perf_counter() - started, reused, stage.mode))
        return RenderExecution(artifacts, tuple(timings))
