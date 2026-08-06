"""Render the motion-smear variants as a deterministic comparison sheet."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from render_smoke_fixture import build_frames

from chronophoto.processing import ComposeSettings, compose_sequence


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def main() -> int:
    output = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/verification/smear-variants.png")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = build_frames()
    variants = (
        ("NONE", "none"),
        ("PHOTOGRAPHIC STRETCH", "photographic"),
        ("DENSE CLONED COPIES", "dense_clones"),
    )
    tile_width, tile_height, title_height = 640, 360, 42
    row_count = (len(variants) + 1) // 2
    sheet = Image.new(
        "RGB",
        (tile_width * 2, (tile_height + title_height) * row_count),
        (11, 11, 11),
    )
    font = _font()

    for index, (label, style) in enumerate(variants):
        result, _ = compose_sequence(
            frames,
            ComposeSettings(
                threshold=16,
                feather=4,
                background="median",
                overlap="newest",
                smear_style=style,
            ),
        )
        Image.fromarray(result).save(output.parent / f"smear-{style}.png")
        tile = Image.fromarray(result).resize(
            (tile_width, tile_height),
            Image.Resampling.LANCZOS,
        )
        left = index % 2 * tile_width
        top = index // 2 * (tile_height + title_height)
        sheet.paste(tile, (left, top + title_height))
        draw = ImageDraw.Draw(sheet)
        draw.text((left + 14, top + 11), label, fill=(232, 232, 232), font=font)

    sheet.save(output)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
