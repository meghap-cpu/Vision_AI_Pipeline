"""When a check is active: time windows such as PPE 06:00-18:00 Mon-Sat. Several windows are allowed; a window
crossing midnight (22:00-06:00) belongs to the day it starts. No schedule = always active. Times are the node's
local time, or the given timezone (e.g. "Asia/Kolkata")."""
from datetime import datetime
from zoneinfo import ZoneInfo

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _minutes(hhmm):
    h, m = str(hhmm).split(":")
    return int(h) * 60 + int(m)


class Schedule:
    def __init__(self, windows=None, timezone=None):
        self.tz = ZoneInfo(timezone) if timezone else None
        self.windows = [({DAYS.index(d[:3].lower()) for d in w.get("days", DAYS)}, _minutes(w["from"]), _minutes(w["to"]))
                        for w in (windows or [])]

    def active(self, ts):
        if not self.windows:
            return True
        now = datetime.fromtimestamp(ts, self.tz) if self.tz else datetime.fromtimestamp(ts)
        mins, day = now.hour * 60 + now.minute, now.weekday()
        for days, start, end in self.windows:
            if start <= end:
                if day in days and start <= mins < end:
                    return True
            elif (day in days and mins >= start) or ((day - 1) % 7 in days and mins < end):  # crosses midnight
                return True
        return False
