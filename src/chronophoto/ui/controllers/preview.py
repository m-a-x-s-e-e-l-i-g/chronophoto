class PreviewController:
    """Track preview revisions without coupling stale-work policy to widgets."""

    def __init__(self) -> None:
        self.pending = False
        self.revision = 0
        self.dirty = False

    def mark_dirty(self) -> int:
        self.revision += 1
        self.dirty = True
        return self.revision

    def mark_current(self) -> None:
        self.dirty = False
