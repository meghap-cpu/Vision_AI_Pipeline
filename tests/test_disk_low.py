"""disk_low opens once and closes with disk_ok, with a 0.5 GB margin. Run: python3 -m tests.test_disk_low"""
import collections, os, tempfile

import vp.retention as retention
from vp.outbox import Outbox

GB = 1e9


def run():
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "snapshots"))
    box = Outbox(os.path.join(d, "outbox.db"), "test-node", "camp")
    r = retention.Retention(box, os.path.join(d, "snapshots"), {"disk_warn_free_gb": 2.0}, every_s=0)
    Usage = collections.namedtuple("Usage", "total used free")
    real = retention.shutil.disk_usage
    steps = [(5.0, "plenty"), (1.5, "drops under 2 GB"), (1.0, "still low"), (2.2, "back over 2 GB, inside the margin"),
             (3.0, "back over 2.5 GB"), (4.0, "still fine"), (1.2, "low again")]
    try:
        for n, (free, what) in enumerate(steps):
            retention.shutil.disk_usage = lambda path, f=free: Usage(100 * GB, 0, f * GB)
            before = box._db().execute("SELECT count(*) FROM events").fetchone()[0]
            r.run_once(now=1000.0 + 300 * n)
            new = box._db().execute("SELECT class, severity, incident_id FROM events ORDER BY rowid LIMIT -1 OFFSET ?", (before,)).fetchall()
            print(f"free {free:3.1f} GB  {what:34s} -> {new}")
    finally:
        retention.shutil.disk_usage = real
    rows = box._db().execute("SELECT class, severity, incident_id FROM events ORDER BY rowid").fetchall()
    got = [(c, s) for c, s, _ in rows]
    assert got == [("disk_low", "warning"), ("disk_ok", "info"), ("disk_low", "warning")], f"unexpected: {got}"
    assert rows[0][2] == rows[1][2] != rows[2][2], "low and ok share one incident; the next drop is a new one"
    print("PASS: one disk_low per drop, no repeats, disk_ok only past 2.5 GB and in the same incident")


if __name__ == "__main__":
    run()
