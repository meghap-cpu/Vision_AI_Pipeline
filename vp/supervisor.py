"""Keeping the node's own loops alive, kept apart from GStreamer so it can be tested.

ErrorCounter: runs a piece of work and never lets its error escape. In GLib a timer whose callback raises is removed
  for good, and a probe that raises breaks the video thread, so every timer and probe goes through it. Errors are
  counted per place, and the first and every 100th are logged.
RestartPolicy: when a crashed plugin worker may start again. At once the first time, then after 5, 10, 20, 40 and
  then every 60 s, so a worker that crashes at start-up can't spin; a worker that has run stable_s without crashing
  starts the sequence over."""
import time
from collections import Counter


class ErrorCounter:
    def __init__(self, log=print):
        self.counts, self.log = Counter(), log

    def run(self, where, fn, *args, default=None):
        try:
            return fn(*args)
        except Exception as e:
            self.counts[where] += 1
            n = self.counts[where]
            if n == 1 or n % 100 == 0:
                self.log(f"{where} error {n}: {type(e).__name__}: {str(e)[:150]}")
            return default


class RestartPolicy:
    def __init__(self, waits=(0, 5, 10, 20, 40, 60), stable_s=300.0, clock=time.time):
        self.waits, self.stable_s, self.clock = waits, stable_s, clock
        self.state, self.restarts = {}, Counter()  # use case -> {started, crashes, dead_since}; restarts done

    def check(self, uc, alive):
        """Called every drain with whether the worker is alive; True = restart it now."""
        now = self.clock()
        st = self.state.setdefault(uc, {"started": now, "crashes": 0, "dead_since": None})
        if alive:
            if st["crashes"] and now - st["started"] >= self.stable_s:
                st["crashes"] = 0  # it has run long enough: a later crash starts the waits over
            return False
        if st["dead_since"] is None:  # just found dead
            st["dead_since"], st["crashes"] = now, st["crashes"] + 1
        if now - st["dead_since"] >= self.waits[min(st["crashes"], len(self.waits)) - 1]:
            st["dead_since"], st["started"] = None, now
            self.restarts[uc] += 1
            return True
        return False

    def next_wait(self, uc):
        st = self.state.get(uc) or {"crashes": 0}
        return self.waits[min(st["crashes"] + 1, len(self.waits)) - 1]
