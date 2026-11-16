"""Keeping the node's loops alive (vp/supervisor.py). Run: python3 -m tests.test_supervisor"""
from vp.supervisor import ErrorCounter, RestartPolicy


def test_errors_never_escape():
    logged = []
    e = ErrorCounter(log=logged.append)
    def broken():
        raise RuntimeError("database is locked")
    results = [e.run("drain", broken, default="kept going") for _ in range(250)]
    assert set(results) == {"kept going"} and e.counts["drain"] == 250
    assert len(logged) == 3 and logged[0].startswith("drain error 1:") and logged[1].startswith("drain error 100:")
    assert e.run("report", lambda x: x * 2, 21) == 42, "work that succeeds returns its result"
    print("ok  250 errors caught and counted, only 3 log lines (1st, 100th, 200th); good work passes through")


def test_crash_loop_slows_down():
    now = [0.0]
    p = RestartPolicy(clock=lambda: now[0])
    alive, restarts = False, []  # a worker that dies at once, every time it starts
    while now[0] < 200:
        if p.check("ppe", alive):
            restarts.append(now[0])
        now[0] += 0.5  # drain runs every 0.5 s
    gaps = [round(b - a, 1) for a, b in zip(restarts, restarts[1:])]
    assert restarts[0] == 0.0, "the first crash restarts at once"
    assert gaps[:5] == [5.5, 10.5, 20.5, 40.5, 60.5], gaps
    print(f"ok  crash loop: {len(restarts)} restarts in 200 s (was 400 before), gaps {gaps}")


def test_stable_worker_starts_over():
    now = [0.0]
    p = RestartPolicy(clock=lambda: now[0])
    assert p.check("crowd", False) is True       # crash 1: at once
    now[0] = 0.5; p.check("crowd", False)         # crash 2 found: waits 5 s
    now[0] = 6.0; assert p.check("crowd", False) is True
    for t in range(7, 400):                       # then runs fine for over 5 min
        now[0] = t; assert p.check("crowd", True) is False
    now[0] = 400; assert p.check("crowd", False) is True, "after running stable, a crash restarts at once again"
    assert p.restarts["crowd"] == 3
    print("ok  a worker that ran 5 min without crashing gets an immediate restart again")


def run():
    test_errors_never_escape(); test_crash_loop_slows_down(); test_stable_worker_starts_over()
    print("PASS: errors never escape the loops; crash loops slow down; stable workers start over")


if __name__ == "__main__":
    run()
