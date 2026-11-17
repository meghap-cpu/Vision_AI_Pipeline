"""Camera watchdog (start-up grace, steady-video rule) and health report (one failing value). Run:
python3 -m tests.test_watchdog_health"""
from vp.health import HealthReporter
from vp.watchdog import CameraWatchdog


class Box:
    def __init__(self): self.events = []
    def add(self, ev): self.events.append(ev)


def test_slow_start_is_not_offline():
    now = [0.0]; box = Box()
    w = CameraWatchdog(["cam00", "cam01"], box, offline_after_s=15, startup_grace_s=60, clock=lambda: now[0])
    while now[0] < 120:
        if now[0] >= 45: w.seen(0)          # cam00 connects late (45 s), like the 16th camera of a busy jetson
        w.check(); now[0] += 0.5            # cam01 never connects
    got = [(e["camera_id"], e["class"]) for e in box.events]
    assert got == [("cam01", "camera_offline")], got
    assert box.events[0]["ts"] >= 60, "a never-seen camera is reported once the grace period is over"
    print("ok  a camera connecting after 45 s raises nothing; one that never connects is offline after 60 s")


def test_flaky_camera_is_one_incident():
    now = [0.0]; box = Box()
    w = CameraWatchdog(["cam05"], box, offline_after_s=15, startup_grace_s=0, steady_s=10, clock=lambda: now[0])
    def run(until, sending):
        while now[0] < until:
            if sending: w.seen(0)
            w.check(); now[0] += 0.5
    run(10, True); run(30, False)           # drops: offline
    for _ in range(5):                      # then 3 s of video, 10 s gone, five times (a bad cable)
        run(now[0] + 3, True); run(now[0] + 10, False)
    run(now[0] + 30, True)                  # finally steady
    got = [e["class"] for e in box.events]
    assert got == ["camera_offline", "camera_online"], got
    assert box.events[0]["incident"] == box.events[1]["incident"]
    print(f"ok  a flaky camera (5 blips) is one incident: offline, then online only after 10 s of steady video "
          f"(offline for {box.events[1]['extra']['offline_for_s']} s)")


def test_health_report_survives_a_failing_value():
    def broken():
        raise RuntimeError("database is locked")
    r = HealthReporter.__new__(HealthReporter)  # no thread, no hub: just the report
    r.node_id, r.site_id, r.started, r.data_dir, r.gpu_load, r.temps = "pilot-01", "camp", 0, "/", None, {}
    r.status = {"cameras_fps": lambda: {"cam00": 5.0}, "unsent_alerts": broken, "outbox_mb": lambda: 12.5}
    h = r.report()
    assert h["cameras_fps"] == {"cam00": 5.0} and h["outbox_mb"] == 12.5 and "unsent_alerts" not in h
    assert h["health_errors"] == ["unsent_alerts"] and "memory_used_pct" in h and "disk_free_gb" in h
    print(f"ok  one failing value: the report still goes, with health_errors {h['health_errors']}")


def run():
    test_slow_start_is_not_offline(); test_flaky_camera_is_one_incident(); test_health_report_survives_a_failing_value()
    print("PASS: no false offline at start-up, flaky cameras are one incident, health reports always go")


if __name__ == "__main__":
    run()
