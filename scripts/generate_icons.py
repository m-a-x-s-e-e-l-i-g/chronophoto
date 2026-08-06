from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANVAS_SIZE = 1024


def _font_path() -> Path:
    override = os.environ.get("CHRONOPHOTO_ICON_FONT")
    candidates = (
        Path(override) if override else None,
        Path("C:/Windows/Fonts/consolab.ttf"),
        Path("/System/Library/Fonts/SFNSMono.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
        PROJECT_ROOT / "src/chronophoto/assets/fonts/BarlowCondensed-SemiBold.ttf",
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError("No suitable font found for the Chronophoto icon")


def create_master_icon() -> Image.Image:
    icon = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    tile = Image.new("RGBA", icon.size, (0, 0, 0, 0))
    ImageDraw.Draw(tile).rounded_rectangle(
        (18, 18, CANVAS_SIZE - 18, CANVAS_SIZE - 18),
        radius=210,
        fill=(8, 8, 8, 255),
    )
    icon.alpha_composite(tile)

    font = ImageFont.truetype(str(_font_path()), 800)
    # Older copies fall back and down, echoing Chronophoto's temporal layering.
    copies = (
        (118, 160, 34),
        (196, 153, 54),
        (274, 146, 82),
        (352, 139, 124),
        (430, 132, 255),
    )
    for x, y, opacity in copies:
        layer = Image.new("RGBA", icon.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((x, y), "C", font=font, fill=(255, 255, 255, opacity))
        icon.alpha_composite(layer)
    return icon


def main() -> None:
    master = create_master_icon()
    outputs = {
        "runtime": PROJECT_ROOT / "src/chronophoto/assets/chronophoto-icon.png",
        "windows": PROJECT_ROOT / "packaging/windows/chronophoto.ico",
        "macos": PROJECT_ROOT / "packaging/macos/chronophoto.icns",
        "linux": PROJECT_ROOT / "packaging/linux/chronophoto.png",
    }
    for output in outputs.values():
        output.parent.mkdir(parents=True, exist_ok=True)

    master.save(outputs["runtime"], optimize=True)
    master.save(
        outputs["windows"],
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    master.save(outputs["macos"], format="ICNS")
    master.save(outputs["linux"], optimize=True)

    for platform, output in outputs.items():
        print(f"{platform}: {output}")


if __name__ == "__main__":
    main()
