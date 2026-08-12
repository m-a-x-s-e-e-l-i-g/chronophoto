from pathlib import Path

import chronophoto.processing.backends as backends
from chronophoto.processing.backends import (
    VideoPipelineCapabilities,
    register_complete_hardware_pipeline,
    select_complete_hardware_pipeline,
)


class _Backend:
    name = "test"

    def __init__(self, *, complete: bool) -> None:
        self.complete = complete

    def capabilities(self, source_path: str | Path) -> VideoPipelineCapabilities:
        del source_path
        return VideoPipelineCapabilities(
            platform="test",
            decoder="GPU decode",
            hardware_decode_candidate="GPU",
            compositor="GPU composite",
            encoder="GPU encode",
            zero_copy_available=self.complete,
            zero_copy_reason="complete" if self.complete else "CPU transfer remains",
        )


def test_hardware_selection_rejects_partial_pipeline(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(backends, "_COMPLETE_HARDWARE_PIPELINES", [])
    register_complete_hardware_pipeline(_Backend(complete=False))

    assert select_complete_hardware_pipeline("source.mp4") is None


def test_hardware_selection_accepts_only_confirmed_zero_copy(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(backends, "_COMPLETE_HARDWARE_PIPELINES", [])
    partial = _Backend(complete=False)
    complete = _Backend(complete=True)
    register_complete_hardware_pipeline(partial)
    register_complete_hardware_pipeline(complete)

    assert select_complete_hardware_pipeline("source.mp4") is complete
