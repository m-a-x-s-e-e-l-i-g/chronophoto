from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import av


@dataclass(slots=True, frozen=True)
class VideoPipelineCapabilities:
    platform: str
    decoder: str
    hardware_decode_candidate: str | None
    compositor: str
    encoder: str
    zero_copy_available: bool
    zero_copy_reason: str


@dataclass(slots=True, frozen=True)
class RenderTelemetry:
    decoder: str
    compositor: str
    encoder: str
    rendered_frames: int
    seconds: float
    render_fps: float
    estimated_peak_bytes: int
    active_window_peak: int
    zero_copy: bool
    bottleneck: str

    def as_text(self) -> str:
        memory_mib = self.estimated_peak_bytes / (1024 * 1024)
        return "\n".join(
            (
                f"Decoder: {self.decoder}",
                f"Compositor: {self.compositor}",
                f"Encoder: {self.encoder}",
                f"Throughput: {self.render_fps:.1f} fps ({self.rendered_frames} frames)",
                f"Estimated peak render memory: {memory_mib:.0f} MiB",
                f"Peak trail window: {self.active_window_peak} frames",
                f"Zero-copy: {'yes' if self.zero_copy else 'no'}",
                f"Likely bottleneck: {self.bottleneck}",
            )
        )


@runtime_checkable
class CompleteHardwareVideoPipeline(Protocol):
    """Optional backend contract for one end-to-end GPU surface pipeline.

    Implementations must keep frames on compatible hardware surfaces from
    decode through composition and encode. A decoder or encoder in isolation
    must not register itself here.
    """

    name: str

    def capabilities(self, source_path: str | Path) -> VideoPipelineCapabilities: ...


_COMPLETE_HARDWARE_PIPELINES: list[CompleteHardwareVideoPipeline] = []


@dataclass(slots=True, frozen=True)
class NvidiaAccelerationSetup:
    gpu_detected: bool
    ready: bool
    detail: str


def _nvidia_device_count() -> int:
    """Ask the NVIDIA driver directly; codec presence does not prove GPU hardware."""

    library_name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    try:
        cuda = ctypes.CDLL(library_name)
        if cuda.cuInit(0) != 0:
            return 0
        count = ctypes.c_int()
        if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return 0
        return max(0, count.value)
    except (AttributeError, OSError):
        return 0


def nvidia_acceleration_setup() -> NvidiaAccelerationSetup:
    """Report whether the optional qualified native NVIDIA backend is available."""

    if _nvidia_device_count() < 1:
        return NvidiaAccelerationSetup(False, False, "No NVIDIA CUDA GPU detected")
    from chronophoto.processing.nvidia_backend import BUNDLED_NVIDIA_PIPELINE

    ready, detail = BUNDLED_NVIDIA_PIPELINE.probe()
    return NvidiaAccelerationSetup(True, ready, detail)


def register_complete_hardware_pipeline(backend: CompleteHardwareVideoPipeline) -> None:
    """Register an optional backend only when it implements the full contract."""

    if not isinstance(backend, CompleteHardwareVideoPipeline):
        raise TypeError("hardware backend does not implement the complete pipeline contract")
    _COMPLETE_HARDWARE_PIPELINES.append(backend)


def select_complete_hardware_pipeline(
    source_path: str | Path,
) -> CompleteHardwareVideoPipeline | None:
    """Return only a backend that confirms an end-to-end zero-copy route."""

    for backend in _COMPLETE_HARDWARE_PIPELINES:
        capabilities = backend.capabilities(source_path)
        if capabilities.zero_copy_available:
            return backend
    return None


def _codec_available(name: str, mode: str) -> bool:
    try:
        av.codec.Codec(name, mode)
    except (av.codec.codec.UnknownCodecError, av.error.FFmpegError):
        return False
    return True


def probe_video_pipeline(source_path: str | Path) -> VideoPipelineCapabilities:
    """Report only complete capabilities; codec presence alone is not zero-copy."""

    with av.open(str(source_path)) as container:
        decoder = container.streams.video[0].codec_context.name
    hardware_decode_candidate = None
    encoder = "CPU H.264"
    reason = "No compatible hardware decode/process/encode surface interop is installed"
    if sys.platform == "darwin":
        if _codec_available("h264_videotoolbox", "w"):
            encoder = "Apple VideoToolbox"
        hardware_decode_candidate = "VideoToolbox"
        reason = "PyAV does not expose a Metal/VideoToolbox zero-copy compositor in this build"
    elif _codec_available("h264_nvenc", "w"):
        encoder = "NVIDIA NVENC"
        if _codec_available("h264_cuvid", "r") or _codec_available("hevc_cuvid", "r"):
            hardware_decode_candidate = "NVIDIA CUVID"
        reason = "CUDA decode and NVENC exist, but no native zero-copy CUDA backend is bundled"
    return VideoPipelineCapabilities(
        platform=sys.platform,
        decoder=f"FFmpeg {decoder} (CPU frames)",
        hardware_decode_candidate=hardware_decode_candidate,
        compositor="Incremental 8-bit alpha (CPU/Pillow)",
        encoder=encoder,
        zero_copy_available=False,
        zero_copy_reason=reason,
    )
