from __future__ import annotations

import json
import wave
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import av
import numpy as np
from PIL import Image

import chronophoto.processing.resolve_export as resolve_export_module
from chronophoto.processing import (
    ComposeCache,
    ComposeSettings,
    EffectKeyframe,
    EffectTrack,
    available_resolve_package_directory,
    build_export_layers,
    write_resolve_package,
)


def _video_fixture() -> tuple[list[np.ndarray], ComposeCache]:
    background = np.full((24, 40, 3), (25, 50, 90), dtype=np.uint8)
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index, left in enumerate((3, 13, 23)):
        frame = background.copy()
        frame[7:18, left : left + 10] = (220 - index * 20, 50 + index * 60, 30)
        mask = np.zeros((24, 40), dtype=np.uint8)
        mask[7:18, left : left + 10] = 255
        frames.append(frame)
        masks.append(mask)
    return frames, ComposeCache(background, masks)


def test_resolve_package_writes_portable_timeline_manifest_and_alpha_media(
    monkeypatch,
    tmp_path: Path,
) -> None:
    frames, cache = _video_fixture()
    keyframes = (EffectKeyframe(0.0, 100.0), EffectKeyframe(1.0, 100.0))
    settings = ComposeSettings(
        overlap="oldest",
        trail_effect_tracks=(EffectTrack("blend_mode", keyframes, option="multiply"),),
    )
    source = tmp_path / "Jump & Grüße.mp4"
    source.write_bytes(b"original-video")
    destination = available_resolve_package_directory(tmp_path, source.stem)

    def fake_audio(output_path, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        path = Path(output_path).with_suffix(".wav")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(48_000)
            output.writeframes(b"\0\0\0\0" * 128)
        return path

    monkeypatch.setattr(
        "chronophoto.processing.resolve_export.write_selected_audio",
        fake_audio,
    )

    result = write_resolve_package(
        destination,
        source_path=source,
        frames=frames,
        cache=cache,
        settings=settings,
        pose_indices=[0, 2],
        pose_labels=["Start & low", "Einde — hoog"],
        effect_progress=[0.0, 0.5, 1.0],
        timestamps=[1.0, 1.5, 2.0],
        start=1.0,
        end=2.5,
        frame_rate=0.0,
        trail_duration=0.75,
        pixel_aspect_ratio=(4, 3),
    )

    assert result.directory == destination
    assert result.has_audio
    assert result.duration == 1.5
    assert (destination / "Media/background.png").is_file()
    packaged_original = destination / "Media/original/Jump & Grüße.mp4"
    assert packaged_original.read_bytes() == b"original-video"
    alpha_video = destination / "Media/trail/trail-alpha.mov"
    assert alpha_video.is_file()
    assert not list((destination / "Media/trail").glob("trail-*.png"))
    assert not (destination / "Media/poses").exists()
    assert not (destination / "Media/composite-reference.mp4").exists()
    with av.open(str(alpha_video)) as container:
        decoded_trails = [frame.to_ndarray(format="rgba") for frame in container.decode(video=0)]
    assert len(decoded_trails) == 3
    second_trail = decoded_trails[1]
    expected_trail = build_export_layers(
        frames[:2],
        cache.masks[:2],
        cache.background,
        settings,
        [0.0, 0.5],
    ).combined_poses
    alpha_delta = np.abs(second_trail[..., 3].astype(int) - expected_trail[..., 3].astype(int))
    assert np.max(alpha_delta) <= 1
    visible = expected_trail[..., 3] > 0
    rgb_delta = np.abs(
        second_trail[..., :3][visible].astype(int) - expected_trail[..., :3][visible].astype(int)
    )
    assert np.max(rgb_delta) <= 8

    timeline_text = result.timeline.read_text(encoding="utf-8")
    root = ET.fromstring(timeline_text.split("<!DOCTYPE fcpxml>\n", 1)[1])
    assert root.attrib["version"] == "1.10"
    library = root.find("./library")
    assert library is not None
    assert library.attrib == {"colorProcessing": "standard"}
    assert root.find("./event") is None
    sequence = root.find("./library/event/project/sequence")
    assert sequence is not None
    assert sequence.attrib["duration"] == "3/2s"
    assert root.find("./resources/format").attrib == {
        "id": "r1",
        "name": "Chronophoto Resolve Timeline",
        "frameDuration": "1/2s",
        "width": "40",
        "height": "24",
        "colorSpace": "1-1-1 (Rec. 709)",
        "paspH": "4",
        "paspV": "3",
    }
    media_urls = [element.attrib["src"] for element in root.findall("./resources/asset/media-rep")]
    assert media_urls
    expected_media_prefix = destination.resolve().as_uri() + "/Media/"
    assert packaged_original.resolve().as_uri() in media_urls
    assert all(url.startswith(expected_media_prefix) and "\\" not in url for url in media_urls)
    assert all("-writing-" not in url for url in media_urls)
    assert "%20" in media_urls[0] and "%C3%BC" in media_urls[0]
    media_paths = [Path(url2pathname(urlparse(url).path)) for url in media_urls]
    assert all(path.is_file() for path in media_paths)

    primary_clips = sequence.findall("./spine/asset-clip")
    assert [clip.attrib["name"] for clip in primary_clips] == ["V2 — Masks / trail"]
    assert [clip.attrib["offset"] for clip in primary_clips] == ["0s"]
    assert primary_clips[0].attrib["duration"] == "3/2s"
    assert all("lane" not in clip.attrib for clip in primary_clips)

    timeline_clips = sequence.findall(".//asset-clip")
    resource_refs = {asset.attrib["id"] for asset in root.findall("./resources/asset")}
    assert {clip.attrib["ref"] for clip in timeline_clips} == resource_refs

    clips = primary_clips[0].findall("./asset-clip")
    background = next(clip for clip in clips if clip.attrib["name"] == "V1 — Background")
    assert background.attrib["lane"] == "-1"
    assert background.attrib["duration"] == "3/2s"
    original = next(clip for clip in clips if clip.attrib["name"].startswith("V3 — Original"))
    assert original.attrib == {
        "name": "V3 — Original Video (disabled)",
        "ref": original.attrib["ref"],
        "lane": "1",
        "offset": "0s",
        "start": "1s",
        "duration": "3/2s",
        "srcEnable": "video",
        "enabled": "0",
    }
    assert any(
        clip.attrib.get("lane") == "-2" and clip.attrib["name"] == "A1 — Source Audio"
        for clip in clips
    )
    assert len(clips) == 3

    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["source"]["selected_in_seconds"] == 1.0
    assert manifest["interchange"]["media_url_mode"] == "absolute_file"
    assert manifest["timeline"]["frame_rate"] == 2.0
    assert manifest["timeline"]["pixel_aspect_ratio"] == "4:3"
    assert manifest["timeline"]["tracks"][:3] == [
        {"lane": "V1", "name": "Background", "enabled": True},
        {"lane": "V2", "name": "Masks / trail", "enabled": True},
        {"lane": "V3", "name": "Original video", "enabled": False},
    ]
    assert manifest["media"]["audio"] == "Media/audio/source-audio.wav"
    assert manifest["media"]["poses"] == []
    assert manifest["media"]["reference"] is None
    assert manifest["media"]["original"] == "Media/original/Jump & Grüße.mp4"
    assert manifest["media"]["trail_representation"] == "alpha_video:qtrle"
    assert "multiply" in manifest["round_trip"]["unsupported_mappings"][0]
    assert available_resolve_package_directory(tmp_path, source.stem).name.endswith("-2")


def test_photo_resolve_package_holds_each_pose_on_ordered_editable_tracks(tmp_path: Path) -> None:
    frames, cache = _video_fixture()
    source = tmp_path / "photo stack 01.png"
    destination = available_resolve_package_directory(tmp_path, source.stem)

    result = write_resolve_package(
        destination,
        source_path=source,
        frames=frames,
        cache=cache,
        settings=ComposeSettings(overlap="newest"),
        pose_indices=[0, 1, 2],
        pose_labels=["One", "Two", "Three"],
        effect_progress=[0.0, 0.5, 1.0],
        timestamps=None,
        start=0.0,
        end=0.0,
        frame_rate=0.0,
        trail_duration=0.0,
    )

    timeline_xml = result.timeline.read_text(encoding="utf-8")
    root = ET.fromstring(timeline_xml.split("<!DOCTYPE fcpxml>\n", 1)[1])
    clips = root.findall("./library/event/project/sequence/spine/asset-clip/asset-clip")
    subject = next(clip for clip in clips if clip.attrib["name"] == "V2 — Subject")
    poses = [clip for clip in clips if "— Pose " in clip.attrib["name"]]
    assert subject.attrib["enabled"] == "0"
    assert [clip.attrib["lane"] for clip in poses] == ["2", "3", "4"]
    assert all(clip.attrib["duration"] == "5s" and clip.attrib["enabled"] == "1" for clip in poses)
    assert result.frame_rate == 30.0
    assert result.duration == 5.0
    assert not result.has_audio
    assert not (destination / "Media/audio").exists()


def test_video_resolve_package_omits_pose_and_reference_renders(
    monkeypatch,
    tmp_path: Path,
) -> None:
    frames, cache = _video_fixture()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original-video")
    destination = tmp_path / "three-track-chronophoto-resolve"
    monkeypatch.setattr(
        "chronophoto.processing.resolve_export.write_selected_audio",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "chronophoto.processing.resolve_export.build_export_layers",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("video exports must not build an unused combined pose layer")
        ),
    )

    write_resolve_package(
        destination,
        source_path=source,
        frames=frames,
        cache=cache,
        settings=ComposeSettings(),
        pose_indices=[0, 1, 2],
        pose_labels=["One", "Two", "Three"],
        effect_progress=[0.0, 0.5, 1.0],
        timestamps=[0.0, 0.5, 1.0],
        start=0.0,
        end=1.5,
        frame_rate=2.0,
        trail_duration=0.75,
    )

    assert not (destination / "Media/poses").exists()
    assert not (destination / "Media/composite-reference.mp4").exists()
    assert (destination / "Media/trail/trail-alpha.mov").is_file()
    root = ET.fromstring(
        (destination / "source.fcpxml")
        .read_text(encoding="utf-8")
        .split(
            "<!DOCTYPE fcpxml>\n",
            1,
        )[1]
    )
    clips = root.findall("./library/event/project/sequence/spine/asset-clip")
    assert len(clips) == 1
    nested = clips[0].findall("./asset-clip")
    assert [clip.attrib["name"] for clip in nested] == [
        "V1 — Background",
        "V3 — Original Video (disabled)",
    ]


def test_video_resolve_package_falls_back_to_png_clips_without_alpha_encoder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    frames, cache = _video_fixture()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original-video")
    destination = tmp_path / "fallback-chronophoto-resolve"
    monkeypatch.setattr(resolve_export_module, "_alpha_video_encoder", lambda: None)
    monkeypatch.setattr(resolve_export_module, "write_selected_audio", lambda *a, **k: None)

    result = write_resolve_package(
        destination,
        source_path=source,
        frames=frames,
        cache=cache,
        settings=ComposeSettings(),
        pose_indices=[0, 1, 2],
        pose_labels=["One", "Two", "Three"],
        effect_progress=[0.0, 0.5, 1.0],
        timestamps=[0.0, 0.5, 1.0],
        start=0.0,
        end=1.5,
        frame_rate=2.0,
        trail_duration=0.75,
    )

    assert len(list((destination / "Media/trail").glob("trail-*.png"))) == 3
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["media"]["trail_representation"] == "png_clip_fallback"


def test_incremental_trail_compositor_is_pixel_exact_for_sliding_windows(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(12)
    poses: list[Image.Image] = []
    for index in range(7):
        pixels = np.zeros((48, 72, 4), dtype=np.uint8)
        left = 4 + index * 7
        pixels[9:39, left : left + 18, :3] = rng.integers(
            0,
            256,
            (30, 18, 3),
            dtype=np.uint8,
        )
        pixels[9:39, left : left + 18, 3] = rng.integers(
            1,
            256,
            (30, 18),
            dtype=np.uint8,
        )
        poses.append(Image.fromarray(pixels))

    windows = [
        (0,),
        (0, 1),
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
        (2, 3, 4, 5),
        (4, 5, 6),
    ]
    for overlap in ("newest", "oldest"):
        jobs = iter(
            (
                tmp_path / f"{overlap}-{frame_index}.png",
                tuple((index, poses[index]) for index in window),
            )
            for frame_index, window in enumerate(windows)
        )
        actual = list(
            resolve_export_module._compose_incremental_trail_frames(
                jobs,
                overlap=overlap,
            )
        )
        for (_path, image), window in zip(actual, windows, strict=True):
            stack = window if overlap == "newest" else tuple(reversed(window))
            _expected_path, expected = resolve_export_module._compose_trail_frame(
                (tmp_path / "expected.png", tuple(poses[index] for index in stack))
            )
            assert np.array_equal(np.asarray(image), np.asarray(expected))


def test_cancelled_resolve_package_removes_partial_output(tmp_path: Path) -> None:
    frames, cache = _video_fixture()
    destination = tmp_path / "cancelled-chronophoto-resolve"

    def cancel_during_media(value: int, message: str) -> None:
        del message
        if value >= 10:
            raise RuntimeError("cancelled for test")

    try:
        write_resolve_package(
            destination,
            source_path=tmp_path / "photos.png",
            frames=frames,
            cache=cache,
            settings=ComposeSettings(),
            pose_indices=[0, 1, 2],
            pose_labels=["One", "Two", "Three"],
            effect_progress=[0.0, 0.5, 1.0],
            timestamps=None,
            start=0.0,
            end=0.0,
            frame_rate=0.0,
            trail_duration=0.0,
            progress=cancel_during_media,
        )
    except RuntimeError as exc:
        assert str(exc) == "cancelled for test"
    else:
        raise AssertionError("Resolve package export was expected to be cancelled")

    assert not destination.exists()
    assert not list(tmp_path.glob(".cancelled-chronophoto-resolve-writing-*"))


def test_resolve_package_publish_retries_a_transient_windows_lock(
    monkeypatch,
    tmp_path: Path,
) -> None:
    temporary = tmp_path / ".package-writing"
    destination = tmp_path / "package"
    temporary.mkdir()
    attempts = 0
    real_replace = resolve_export_module.os.replace

    def transient_replace(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporarily locked")
        real_replace(source, target)

    monkeypatch.setattr(resolve_export_module.os, "replace", transient_replace)
    monkeypatch.setattr(resolve_export_module.time, "sleep", lambda _delay: None)

    resolve_export_module._publish_directory(temporary, destination)

    assert attempts == 3
    assert destination.is_dir()
    assert not temporary.exists()
