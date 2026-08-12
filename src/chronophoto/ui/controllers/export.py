from __future__ import annotations

from chronophoto.processing import ExportKind


class ExportRecipeController:
    EXCLUSIVE = frozenset(("trail_video", "resolve_timeline"))
    LABELS: dict[ExportKind, str] = {
        "composite": "COMPOSITE",
        "combined_poses": "POSES",
        "individual_poses": "SEPARATE POSES",
        "background": "BACKGROUND",
        "trail_video": "TRAIL VIDEO",
        "resolve_timeline": "DAVINCI RESOLVE",
    }
    DESCRIPTIONS: dict[ExportKind, str] = {
        "composite": "Finished composite image",
        "combined_poses": "Combined transparent pose layer",
        "individual_poses": "Separate transparent pose layers",
        "background": "Processed clean plate",
        "trail_video": "Motion-trail MP4 with source audio",
        "resolve_timeline": "Editable DaVinci Resolve package",
    }

    def normalized_after_check(
        self,
        selections: tuple[ExportKind, ...],
        checked: ExportKind,
    ) -> tuple[ExportKind, ...]:
        if checked in self.EXCLUSIVE:
            return (checked,)
        return tuple(item for item in selections if item not in self.EXCLUSIVE)

    def short_summary(self, selections: tuple[ExportKind, ...]) -> str:
        if not selections:
            return "NONE"
        if len(selections) == 1:
            return self.LABELS[selections[0]]
        return f"{len(selections)} SELECTED"

    def description(self, selections: tuple[ExportKind, ...]) -> str:
        if not selections:
            return "Choose at least one output"
        return (
            " + ".join(self.DESCRIPTIONS[item] for item in selections) + " · full source resolution"
        )
