"""Crops a plugin's zone out of a frame on the video thread: small and grayscale, so it is cheap to send."""
import cv2

MAX_WIDTH = 320


def crop_color(surface, zone, max_width=512):
    """The zone in colour (RGB), at most max_width wide: what the VLM sees."""
    h, w = surface.shape[:2]
    x1, y1, x2, y2 = int(zone[0] * w), int(zone[1] * h), int(zone[2] * w), int(zone[3] * h)
    roi = surface[y1:y2, x1:x2]
    scale = min(1.0, max_width / max(1, roi.shape[1]))
    if scale < 1.0:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)


def crop_color(surface, zone, max_width=512):
    """The zone in colour (RGB), at most max_width wide: what the VLM sees."""
    h, w = surface.shape[:2]
    x1, y1, x2, y2 = int(zone[0] * w), int(zone[1] * h), int(zone[2] * w), int(zone[3] * h)
    roi = surface[y1:y2, x1:x2]
    scale = min(1.0, max_width / max(1, roi.shape[1]))
    if scale < 1.0:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)


def crop_zone(surface, zone):
    """surface: the RGBA frame (H, W, 4). zone: x1, y1, x2, y2 as fractions. Returns a small gray image."""
    h, w = surface.shape[:2]
    x1, y1, x2, y2 = int(zone[0] * w), int(zone[1] * h), int(zone[2] * w), int(zone[3] * h)
    roi = surface[y1:y2, x1:x2]
    scale = min(1.0, MAX_WIDTH / max(1, roi.shape[1]))
    if scale < 1.0:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(roi, cv2.COLOR_RGBA2GRAY)
