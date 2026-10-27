"""Use-case name in node.yaml -> plugin class."""
from vp.plugins.crowd import CrowdPlugin
from vp.plugins.fire_smoke import FireSmokePlugin
from vp.plugins.housekeeping import HousekeepingPlugin
from vp.plugins.man_down import ManDownPlugin
from vp.plugins.motion import MotionPlugin
from vp.plugins.ppe import PPEPlugin
from vp.plugins.test_plugin import TestPlugin
from vp.plugins.worker_at_height import WorkerAtHeightPlugin

REGISTRY = {cls.name: cls for cls in (TestPlugin, MotionPlugin, CrowdPlugin, HousekeepingPlugin, PPEPlugin,
                                      ManDownPlugin, FireSmokePlugin, WorkerAtHeightPlugin)}
