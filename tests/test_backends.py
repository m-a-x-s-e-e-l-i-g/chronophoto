from pathlib import Path

import chronophoto.processing.backends as backends
import chronophoto.processing.nvidia_backend as nvidia_backend
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


def test_nvidia_setup_reports_unbundled_native_backend(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(backends, "_nvidia_device_count", lambda: 1)
    monkeypatch.setattr(
        nvidia_backend.BUNDLED_NVIDIA_PIPELINE,
        "probe",
        lambda: (False, "native backend unavailable"),
    )

    setup = backends.nvidia_acceleration_setup()

    assert setup.gpu_detected
    assert not setup.ready
    assert setup.detail == "native backend unavailable"


def test_nvidia_setup_stays_hidden_without_nvidia_hardware(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(backends, "_nvidia_device_count", lambda: 0)

    setup = backends.nvidia_acceleration_setup()

    assert not setup.gpu_detected


def test_nvidia_qualification_requires_equivalence_and_speedup() -> None:
    assert nvidia_backend.NvidiaQualification().passed
    assert not nvidia_backend.NvidiaQualification(mean_absolute_error=2.01).passed
    assert not nvidia_backend.NvidiaQualification(pixel_fraction=0.979).passed
    assert not nvidia_backend.NvidiaQualification(gpu_fps=2.4, cpu_fps=2.05).passed


def test_nvidia_backend_falls_back_before_unsupported_effect_render() -> None:
    from chronophoto.processing.compositor import ComposeSettings
    from chronophoto.processing.effects import EffectKeyframe, EffectTrack

    effect = EffectTrack(
        "opacity",
        (EffectKeyframe(0.0, 100.0), EffectKeyframe(1.0, 100.0)),
    )

    assert nvidia_backend.BUNDLED_NVIDIA_PIPELINE.supports(ComposeSettings(), "off", 1920, 1080)
    assert not nvidia_backend.BUNDLED_NVIDIA_PIPELINE.supports(
        ComposeSettings(trail_effect_tracks=(effect,)), "off", 1920, 1080
    )
