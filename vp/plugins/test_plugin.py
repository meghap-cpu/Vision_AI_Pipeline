from vp.plugin import Plugin, event


class TestPlugin(Plugin):
    """Raises one event per new person track, to exercise the plumbing."""
    name = "test"

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        self.seen = {}  # track_id -> last time seen
        self.forget_after = params.get("forget_after_s", 30)

    def on_frame(self, frame):
        events = []
        for o in (o for o in frame.objects if o.source == 1 and o.class_id == 0):
            if o.track_id not in self.seen:
                events.append(event(self, frame, "new_person", severity="warning", confidence=o.confidence,
                                    track_id=o.track_id, box=o.box, has_pose=o.keypoints is not None))
            self.seen[o.track_id] = frame.ts
        self.seen = {t: s for t, s in self.seen.items() if frame.ts - s < self.forget_after}
        return events
