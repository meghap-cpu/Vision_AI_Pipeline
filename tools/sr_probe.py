"""Feasibility test: can Python trigger DeepStream smart record on nvurisrcbin?
Usage: python3 tools/sr_probe.py A|B   (A: session pointer None; B: a real pointer from ctypes)"""
import ctypes, glob, os, sys, time
import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

variant, out_dir = sys.argv[1], f"/tmp/clips_{sys.argv[1]}"
os.makedirs(out_dir, exist_ok=True)
for f in glob.glob(out_dir + "/*"):
    os.remove(f)
Gst.init(None)
pipe = Gst.Pipeline.new("p")
src = Gst.ElementFactory.make("nvurisrcbin", "src")
for k, v in {"uri": "rtsp://127.0.0.1:8554/cam0", "smart-record": 2, "smart-rec-dir-path": out_dir,
             "smart-rec-file-prefix": "cam00", "smart-rec-cache": 10, "smart-rec-container": 0,
             "smart-rec-default-duration": 10}.items():
    src.set_property(k, v)
mux = Gst.ElementFactory.make("nvstreammux", "mux")
mux.set_property("batch-size", 1); mux.set_property("width", 1280); mux.set_property("height", 720)
mux.set_property("live-source", True)
sink = Gst.ElementFactory.make("fakesink", "sink")
for el in (src, mux, sink):
    pipe.add(el)
mux.link(sink)
sinkpad = mux.request_pad_simple("sink_0")
src.connect("pad-added", lambda s, pad: pad.link(sinkpad) if pad.query_caps(None).get_structure(0).get_name().startswith("video") else None)
session = ctypes.c_uint(0)
done = {}


def on_done(*args):
    done["called"] = True
    return None


try:
    src.connect("sr-done", on_done)
except TypeError as e:
    print("could not connect sr-done:", e)


def trigger():
    print(f"[{variant}] emitting start-sr (5 s before now, 10 s long)", flush=True)
    try:
        if variant == "A":
            src.emit("start-sr", None, 5, 10, None)
        elif variant == "C":  # call GLib's C emit directly, with a real session-id address
            gobj = ctypes.CDLL("libgobject-2.0.so.0")
            gobj.g_type_name_from_instance.restype = ctypes.c_char_p
            gobj.g_signal_emit_by_name.restype = None
            ptr = ctypes.c_void_p.from_address(id(src) + 2 * ctypes.sizeof(ctypes.c_void_p)).value
            kind = gobj.g_type_name_from_instance(ctypes.c_void_p(ptr))
            print(f"[{variant}] C pointer check: element type is {kind!r}", flush=True)
            if kind is None or b"UriSrcBin" not in kind:
                raise RuntimeError("pointer check failed; not emitting")
            gobj.g_signal_emit_by_name(ctypes.c_void_p(ptr), b"start-sr", ctypes.byref(session),
                                       ctypes.c_uint(5), ctypes.c_uint(10), ctypes.c_void_p(None))
        else:
            src.emit("start-sr", ctypes.addressof(session), 5, 10, None)
        print(f"[{variant}] emit returned; session id = {session.value}", flush=True)
    except Exception as e:
        print(f"[{variant}] emit raised: {type(e).__name__}: {e}", flush=True)
    return False


loop = GLib.MainLoop()
GLib.timeout_add_seconds(15, trigger)
GLib.timeout_add_seconds(35, lambda: loop.quit() or False)
pipe.set_state(Gst.State.PLAYING)
loop.run()
pipe.set_state(Gst.State.NULL)
files = glob.glob(out_dir + "/*")
print(f"[{variant}] sr-done called: {done.get('called', False)}; files: "
      f"{[(os.path.basename(f), os.path.getsize(f)) for f in files] or 'none'}", flush=True)
