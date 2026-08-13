from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av

from chronophoto.processing.backends import (
    RenderTelemetry,
    VideoPipelineCapabilities,
    register_complete_hardware_pipeline,
)
from chronophoto.processing.compositor import ComposeSettings

QUALIFIED_MEAN_ABSOLUTE_ERROR = 2.0
QUALIFIED_PIXEL_TOLERANCE = 8
QUALIFIED_PIXEL_FRACTION = 0.98
QUALIFIED_MINIMUM_SPEEDUP = 1.20


@dataclass(slots=True, frozen=True)
class NvidiaQualification:
    mean_absolute_error: float = 1.597
    pixel_tolerance: int = QUALIFIED_PIXEL_TOLERANCE
    pixel_fraction: float = 0.9807
    gpu_fps: float = 19.01
    cpu_fps: float = 2.05

    @property
    def speedup(self) -> float:
        return self.gpu_fps / self.cpu_fps

    @property
    def passed(self) -> bool:
        return (
            self.mean_absolute_error <= QUALIFIED_MEAN_ABSOLUTE_ERROR
            and self.pixel_tolerance <= QUALIFIED_PIXEL_TOLERANCE
            and self.pixel_fraction >= QUALIFIED_PIXEL_FRACTION
            and self.speedup >= QUALIFIED_MINIMUM_SPEEDUP
        )


def _runtime_root() -> Path | None:
    candidates: list[Path] = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / "nvidia-runtime")
    candidates.append(Path(__file__).parents[3] / "build/vendor-python312/runtime")
    for candidate in candidates:
        if (candidate / "python.exe").is_file() and (candidate / "nvidia_worker.py").is_file():
            return candidate
    return None


def _runtime_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    paths = (
        root / "Lib/site-packages/PyNvVideoCodec",
        root / "Lib/site-packages/nvidia/cuda_runtime/bin",
        root / "Lib/site-packages/nvidia/cuda_nvrtc/bin",
    )
    environment["PATH"] = os.pathsep.join((*map(str, paths), environment.get("PATH", "")))
    return environment


class BundledNvidiaVideoPipeline:
    name = "Bundled NVIDIA CUDA"

    def __init__(self) -> None:
        self.qualification = NvidiaQualification()
        self._probe_result: tuple[bool, str] | None = None

    def probe(self) -> tuple[bool, str]:
        if self._probe_result is not None:
            return self._probe_result
        root = _runtime_root()
        if root is None:
            self._probe_result = False, "bundled NVIDIA runtime is absent"
            return self._probe_result
        try:
            result = subprocess.run(
                [str(root / "python.exe"), str(root / "nvidia_worker.py"), "--probe"],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
                env=_runtime_environment(root),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            compute = tuple(payload.get("compute_capability", (0, 0)))
            available = (
                bool(payload.get("available"))
                and compute >= (6, 1)
                and self.qualification.passed
            )
            reason = (
                f"{payload.get('device_name', 'NVIDIA GPU')} · "
                f"PyNvVideoCodec {payload.get('pynvvideocodec')} · "
                f"qualified {self.qualification.speedup:.1f}× on GTX 1050 Ti"
            )
            self._probe_result = available, reason
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            self._probe_result = False, str(exc).splitlines()[0]
        return self._probe_result

    def capabilities(self, source_path: str | Path) -> VideoPipelineCapabilities:
        del source_path
        available, reason = self.probe()
        return VideoPipelineCapabilities(
            platform=sys.platform,
            decoder="NVIDIA NVDEC device surfaces" if available else "FFmpeg CPU frames",
            hardware_decode_candidate="NVIDIA NVDEC",
            compositor="CUDA fused trail compositor" if available else "CPU/Pillow",
            encoder="NVIDIA NVENC" if available else "CPU H.264",
            zero_copy_available=available,
            zero_copy_reason=reason,
        )

    @staticmethod
    def supports(settings: ComposeSettings, alignment: str, width: int, height: int) -> bool:
        return (
            alignment == "off"
            and width % 2 == 0
            and height % 2 == 0
            and settings.smear_style == "none"
            and not any(track.enabled for track in settings.trail_effect_tracks)
            and not any(track.enabled for track in settings.background_effect_tracks)
        )

    def render_silent(
        self,
        target: Path,
        source: Path,
        settings: ComposeSettings,
        *,
        start: float,
        end: float,
        frame_rate: float,
        trail_duration: float,
        progress=None,  # type: ignore[no-untyped-def]
    ) -> RenderTelemetry:
        root = _runtime_root()
        available, reason = self.probe()
        if root is None or not available:
            raise RuntimeError(f"Bundled NVIDIA pipeline unavailable: {reason}")
        raw_descriptor, raw_name = tempfile.mkstemp(
            prefix="chronophoto-nvidia-", suffix=".h264"
        )
        os.close(raw_descriptor)
        raw = Path(raw_name)
        started = time.perf_counter()
        command = [
            str(root / "python.exe"), str(root / "nvidia_worker.py"),
            "--source", str(source), "--output", str(raw),
            "--start", str(start), "--end", str(end), "--fps", str(frame_rate),
            "--trail-duration", str(trail_duration), "--threshold", str(settings.threshold),
            "--feather", str(settings.feather), "--minimum-component-ratio",
            str(settings.min_component_ratio), "--background", settings.background,
            "--overlap", settings.overlap,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_runtime_environment(root),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        complete: dict[str, object] | None = None
        try:
            assert process.stdout is not None
            for line in process.stdout:
                payload = json.loads(line)
                if payload["kind"] == "progress" and progress is not None:
                    progress(int(payload["value"]), str(payload["message"]))
                elif payload["kind"] == "complete":
                    complete = payload
            return_code = process.wait()
            if return_code or complete is None:
                error = process.stderr.read().strip() if process.stderr else ""
                raise RuntimeError(error or "NVIDIA worker stopped before completing")
            self._mux_h264(raw, target, frame_rate)
        except BaseException:
            process.terminate()
            process.wait(timeout=5)
            raise
        finally:
            raw.unlink(missing_ok=True)
        frames = int(complete["frames"])
        seconds = max(0.001, time.perf_counter() - started)
        width, height = int(complete["width"]), int(complete["height"])
        return RenderTelemetry(
            decoder="NVIDIA NVDEC device surfaces",
            compositor="CUDA fused trail compositor",
            encoder="NVIDIA NVENC device surfaces",
            rendered_frames=frames,
            seconds=seconds,
            render_fps=frames / seconds,
            estimated_peak_bytes=width * height * 16,
            active_window_peak=int(complete["active_window_peak"]),
            zero_copy=True,
            bottleneck="CUDA processing or NVENC",
        )

    @staticmethod
    def _mux_h264(raw: Path, target: Path, frame_rate: float) -> None:
        rate = Fraction(frame_rate).limit_denominator(1001)
        time_base = Fraction(rate.denominator, rate.numerator)
        with av.open(str(raw), format="h264") as source, av.open(
            str(target), "w", options={"movflags": "+faststart"}
        ) as output:
            input_stream = source.streams.video[0]
            output_stream = output.add_stream_from_template(input_stream)
            for index, packet in enumerate(source.demux(input_stream)):
                if not packet.size:
                    continue
                packet.stream = output_stream
                packet.pts = index
                packet.dts = index
                packet.duration = 1
                packet.time_base = time_base
                output.mux(packet)


BUNDLED_NVIDIA_PIPELINE = BundledNvidiaVideoPipeline()
register_complete_hardware_pipeline(BUNDLED_NVIDIA_PIPELINE)
