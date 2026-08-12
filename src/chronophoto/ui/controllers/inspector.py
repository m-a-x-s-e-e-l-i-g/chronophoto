from chronophoto.processing import ComposeSettings


def should_expand_advanced(
    settings: ComposeSettings,
    *,
    alignment: str,
    photo_order: str,
    source_kind: str,
) -> bool:
    default_alignment = "off" if source_kind == "video" else "translation"
    return any(
        (
            settings.threshold != 17,
            settings.feather != 1,
            settings.background != "median",
            settings.overlap != "newest",
            settings.smear_style != "none",
            alignment != default_alignment,
            photo_order != "automatic",
        )
    )
