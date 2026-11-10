"""Schedules: time windows, and the worker pausing and resuming a plugin. Run: python3 -m tests.test_schedule"""
import queue
from datetime import datetime

from vp.plugin import Frame, Obj
from vp.schedule import Schedule
from vp import worker


def ts(y, mo, d, h, mi, s=0):
    return datetime(y, mo, d, h, mi, s).timestamp()  # node local time


def run():
    # 2026-10-26 is a Monday
    day = Schedule([{"days": ["mon", "tue", "wed", "thu", "fri"], "from": "06:00", "to": "18:00"}])
    assert not day.active(ts(2026, 10, 26, 5, 59)) and day.active(ts(2026, 10, 26, 6, 0))
    assert day.active(ts(2026, 10, 26, 17, 59)) and not day.active(ts(2026, 10, 26, 18, 0))
    assert not day.active(ts(2026, 10, 31, 10, 0))  # Saturday
    night = Schedule([{"days": ["fri"], "from": "22:00", "to": "06:00"}])  # Friday night into Saturday
    assert night.active(ts(2026, 10, 30, 23, 0)) and night.active(ts(2026, 10, 31, 5, 0))
    assert not night.active(ts(2026, 10, 31, 7, 0)) and not night.active(ts(2026, 10, 29, 23, 0))
    assert Schedule().active(ts(2026, 10, 31, 3, 0))  # no schedule: always on
    print("time windows: day, edges, weekend, overnight and always-on all correct")

    frames, events = queue.Queue(), queue.Queue()
    person = [Obj(1, 0, 0.9, (0.1, 0.1, 0.2, 0.5), source=1)]
    for n, t in enumerate([ts(2026, 10, 26, 17, 59, 58), ts(2026, 10, 26, 18, 0, 1), ts(2026, 10, 27, 6, 0, 0)]):
        frames.put(Frame("cam01", n, t, person))
    frames.put(None)
    worker.run("test", {"cam01": {"schedule": [{"from": "06:00", "to": "18:00"}]}}, frames, events)
    got = []
    while not events.empty():
        got.append(events.get()["class"])
    print("worker events:", got)
    assert got == ["new_person", "schedule_paused", "schedule_resumed", "new_person"], got
    print("PASS: windows correct; the plugin pauses at 18:00, resumes at 06:00 and starts fresh (same person reported again)")


if __name__ == "__main__":
    run()
