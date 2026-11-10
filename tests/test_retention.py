"""Storage housekeeping on made-up outbox data. Run: python3 -m tests.test_retention"""
import os, tempfile, time

from vp.outbox import Outbox
from vp.retention import Retention

DAY = 86400


def run():
    d = tempfile.mkdtemp()
    snaps = os.path.join(d, "snapshots"); os.makedirs(snaps)
    box, now = Outbox(os.path.join(d, "outbox.db"), "test-node", "camp"), time.time()

    def ev(age_days, sent, snapshot=None):
        eid = box.add({"camera_id": "cam01", "use_case": "test", "class": "x", "severity": "warning", "ts": now})
        with box._db() as db:
            db.execute("UPDATE events SET created = ?, sent = ? WHERE event_id = ?", (now - age_days * DAY, int(sent), eid))
        if snapshot:
            path = os.path.join(snaps, snapshot)
            if not os.path.exists(path):
                open(path, "wb").write(b"\0" * 1_000_000)  # 1 MB
            box.set_snapshot([eid], path)
        return eid

    e1 = ev(3, True, "old_delivered.jpg")       # old, delivered          -> deleted (age)
    ev(3, False, "old_undelivered.jpg")         # old, NOT delivered      -> kept
    ev(0, True, "new_delivered.jpg")            # recent, delivered       -> deleted only to fit the cap
    ev(0, True, "shared.jpg")                   # shared by a delivered...
    ev(0, False, "shared.jpg")                  # ...and an undelivered event -> kept
    e6 = ev(10, True)                           # sent 10 days ago        -> pruned
    r = Retention(box, snaps, {"snapshot_keep_hours": 24, "snapshot_max_gb": 0.0015, "outbox_keep_days": 7}, every_s=0)
    stats = r.run_once(now)
    left = sorted(os.listdir(snaps))
    ids = {e for (e,) in box._db().execute("SELECT event_id FROM events")}
    warn = box._db().execute("SELECT count(*) FROM events WHERE class = 'storage_full'").fetchone()[0]
    e1_snap = box._db().execute("SELECT snapshot FROM events WHERE event_id = ?", (e1,)).fetchone()[0]
    print("files left:", left, "| stats:", stats, "| storage_full warnings:", warn)
    assert left == ["old_undelivered.jpg", "shared.jpg"], left
    assert e6 not in ids and e1 in ids and e1_snap is None
    assert warn == 1
    print("PASS: delivered snapshots removed by age and to fit the cap, undelivered kept, old sent events pruned, "
          "storage_full raised when only undelivered data is left")


if __name__ == "__main__":
    run()
