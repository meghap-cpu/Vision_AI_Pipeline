#!/usr/bin/env bash
# Sets up an edge jetson to run the vision node, the same way as the pilot (pilot-01, Day 12).
# Safe to run again: every step checks first and skips what is already done.
#
# Usage (from the project folder, as the user who will run the node):
#   tools/install_node.sh --check                       report only, change nothing
#   tools/install_node.sh [--node-id ID --hub URL] [--onnx DIR] [--start]
#     --node-id ID   this jetson's name, e.g. edge-02 (rewrites config/bootstrap.yaml)
#     --hub URL      hub address, e.g. http://192.168.0.50:8100
#     --onnx DIR     folder holding yolo26s.onnx, ppe-dyn-rgb01.onnx, pose-rgb01.onnx (copied into models/)
#     --start        start the node at the end
# Engines are always built on the jetson itself (never copied): about 45 min the first time.
# Until production jetsons get SSDs, this replaces the container on edge jetsons.
set -u
cd "$(dirname "$0")/.." || exit 1
PROJECT=$(pwd); ME=$(id -un)
CHECK=0; NODE_ID=""; HUB=""; ONNX_DIR=""; START=0
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1 ;; --start) START=1 ;;
    --node-id) NODE_ID="$2"; shift ;; --hub) HUB="$2"; shift ;; --onnx) ONNX_DIR="$2"; shift ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac; shift
done
if [ -n "$NODE_ID" ] && [ -z "$HUB" ] || [ -z "$NODE_ID" ] && [ -n "$HUB" ]; then echo "give --node-id and --hub together"; exit 2; fi
# PY, TRTEXEC and UNIT can be overridden from the environment (used to test this script off a jetson)
PY=${PY:-/usr/bin/python3}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
PYDS_URL="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_aarch64.whl"
APT_PKGS="python3-gi python3-gst-1.0 python3-numpy python3-yaml python3-pip gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav g++"
FAILS=0
step() { echo; echo "=== $*"; }
ok()   { echo "  ok: $*"; }
todo() { echo "  to do: $*"; }
bad()  { echo "  FAILED: $*"; FAILS=$((FAILS + 1)); }
run()  { if [ $CHECK = 1 ]; then todo "$*"; else "$@"; fi; }

step "1. This jetson matches the pilot"
[ "$(uname -m)" = aarch64 ] && ok "arm64" || { bad "not a jetson (arm64)"; exit 1; }
L4T=$(head -1 /etc/nv_tegra_release 2>/dev/null | sed -E 's/# R([0-9]+) \(release\), REVISION: ([0-9.]+).*/\1.\2/')
[ "${L4T%%.*}" = 36 ] && ok "L4T $L4T (pilot: 36.4.7)" || { bad "L4T $L4T, need JetPack 6 (R36)"; exit 1; }
[ "$L4T" = 36.4.7 ] || echo "  note: L4T differs from the pilot's 36.4.7; engines are built here, so this is usually fine"
grep -q "Version: 7.1" /opt/nvidia/deepstream/deepstream/version 2>/dev/null && ok "DeepStream 7.1" || { bad "DeepStream 7.1 not installed"; exit 1; }
$PY -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))' && ok "system python 3.10" || { bad "/usr/bin/python3 is not 3.10"; exit 1; }
[ -x "$TRTEXEC" ] && ok "trtexec" || { bad "trtexec missing at $TRTEXEC"; exit 1; }

step "2. System packages"
MISSING=$(for p in $APT_PKGS; do dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || echo "$p"; done | xargs)
if [ -z "$MISSING" ]; then ok "all installed"; else run sudo apt-get install -y --no-install-recommends $MISSING || bad "apt install"; fi
$PY -c 'import cv2' 2>/dev/null && ok "OpenCV $($PY -c 'import cv2; print(cv2.__version__)') (JetPack's own)" \
  || run sudo apt-get install -y libopencv-python || bad "OpenCV"

step "3. Python packages (pip --user, numpy stays the system one)"
$PY -c 'import pyds' 2>/dev/null && ok "pyds" || run $PY -m pip install --user --no-deps "$PYDS_URL" || bad "pyds"
[ "$($PY -c 'import yaml; print(yaml.__version__)' 2>/dev/null)" = 6.0.3 ] && ok "PyYAML 6.0.3" \
  || run $PY -m pip install --user --no-deps "PyYAML==6.0.3" || bad "PyYAML"
NP=$($PY -c 'import numpy; print(numpy.__version__)' 2>/dev/null)
[ "$NP" = 1.21.5 ] && ok "numpy 1.21.5" || echo "  note: numpy $NP (pilot: 1.21.5 from apt; pyds needs 1.x)"
if [ $CHECK = 0 ]; then $PY -c 'import pyds, numpy, cv2, yaml, gi' 2>/dev/null && ok "node imports work (pyds, numpy, cv2, yaml, gi)" || bad "an import fails: $PY -c 'import pyds, numpy, cv2, yaml, gi'"; fi

step "4. Output parsers"
for f in parsers/*.cpp; do
  so="parsers/lib$(basename "$f" .cpp).so"
  if [ -f "$so" ] && [ "$so" -nt "$f" ]; then ok "$so"; continue; fi
  if [ $CHECK = 1 ]; then todo "compile $so"; continue; fi
  g++ -std=c++17 -O2 -shared -fPIC -o "$so.new" "$f" -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda/include \
    && mv "$so.new" "$so" && ok "compiled $so" || bad "compile $so"
done

step "5. Models (ONNX in models/)"
mkdir -p models
for m in yolo26s.onnx ppe-dyn-rgb01.onnx pose-rgb01.onnx; do
  if [ -s "models/$m" ]; then ok "models/$m"
  elif [ -n "$ONNX_DIR" ] && [ -s "$ONNX_DIR/$m" ]; then
    if [ $CHECK = 1 ]; then todo "copy $m from $ONNX_DIR"; else cp "$ONNX_DIR/$m" models/ && ok "copied $m" || bad "copy $m"; fi
  else bad "models/$m missing: pass --onnx DIR"; fi
done

step "6. TensorRT engines (built here, never copied; ~45 min the first time)"
# name | onnx | input | min | opt | max   (same flags as the pilot's build logs)
while IFS='|' read -r ENG ONNX INP MIN OPT MAX; do
  if [ -s "models/$ENG" ]; then ok "models/$ENG"; continue; fi
  if [ $CHECK = 1 ]; then todo "build models/$ENG from $ONNX"; continue; fi
  if [ ! -s "models/$ONNX" ]; then bad "can't build $ENG: models/$ONNX missing"; continue; fi
  echo "  building $ENG (log: models/build_${ENG%.engine}.log) ..."
  (cd models && $TRTEXEC --onnx="$ONNX" --saveEngine="$ENG" --fp16 \
     --minShapes="$INP:$MIN" --optShapes="$INP:$OPT" --maxShapes="$INP:$MAX" > "build_${ENG%.engine}.log" 2>&1)
  grep -q "&&&& PASSED" "models/build_${ENG%.engine}.log" && ok "built $ENG ($(grep -oE 'Engine built in [0-9.]+ sec' "models/build_${ENG%.engine}.log"))" \
    || { rm -f "models/$ENG"; bad "build $ENG: see models/build_${ENG%.engine}.log"; }
done << 'EOF'
yolo26s_b16_fp16.engine|yolo26s.onnx|images|1x3x640x640|8x3x640x640|16x3x640x640
ppe-rgb01_b16_fp16.engine|ppe-dyn-rgb01.onnx|image|1x3x560x560|8x3x560x560|16x3x560x560
pose-rgb01_b32_fp16.engine|pose-rgb01.onnx|image|1x3x256x192|16x3x256x192|32x3x256x192
EOF

step "7. Power mode (pilot: mode 3; clocks are locked by the service at every start)"
MODE=$(sudo nvpmodel -q 2>/dev/null | tail -1)
if [ "$MODE" = 3 ]; then ok "nvpmodel mode 3"
elif [ $CHECK = 1 ]; then todo "set nvpmodel mode 3 (now: $MODE)"
else sudo nvpmodel -m 3 < /dev/null && echo "  set mode 3; if nvpmodel asked for a reboot, reboot after this script" || bad "nvpmodel"; fi

step "8. This jetson's name and hub"
if [ -n "$NODE_ID" ]; then
  if [ $CHECK = 1 ]; then todo "write config/bootstrap.yaml: $NODE_ID, ${HUB%/}"
  else printf '%s\n' '# Bootstrap: who I am and where my settings come from' "node_id: $NODE_ID" \
         "config_url: ${HUB%/}/api/v1/nodes/$NODE_ID/config" > config/bootstrap.yaml && ok "config/bootstrap.yaml: $NODE_ID, ${HUB%/}"; fi
  echo "  note: the hub needs hub/configs/$NODE_ID.yaml for this jetson"
else
  ok "keeping config/bootstrap.yaml ($(grep node_id config/bootstrap.yaml))"
fi

step "9. Service"
UNIT=${UNIT:-/etc/systemd/system/vision-node.service}
WANT=$(sed -e "s|^User=.*|User=$ME|" -e "s|^WorkingDirectory=.*|WorkingDirectory=$PROJECT|" tools/vision-node.service)
if [ -f "$UNIT" ] && [ "$(cat "$UNIT")" = "$WANT" ]; then ok "vision-node service (user $ME, $PROJECT)"
elif [ $CHECK = 1 ]; then todo "install $UNIT (user $ME, $PROJECT)"
else echo "$WANT" | sudo tee "$UNIT" > /dev/null && sudo systemctl daemon-reload && sudo systemctl enable vision-node -q && ok "installed and enabled"; fi

step "10. Tests"
if [ $CHECK = 1 ]; then todo "run tests/"; else
  for t in tests/test_*.py; do m=$(basename "$t" .py)
    $PY -m "tests.$m" > "/tmp/t_$m.log" 2>&1 && echo "  PASS $m" || bad "$m (see /tmp/t_$m.log)"
  done
fi

if [ $START = 1 ] && [ $CHECK = 0 ] && [ $FAILS = 0 ]; then step "11. Start"; sudo systemctl restart vision-node && ok "started: journalctl -u vision-node -f"; fi
echo
if [ $CHECK = 1 ]; then echo "check only: nothing was changed ($FAILS problem(s))"
elif [ $FAILS = 0 ]; then echo "done: this jetson is set up like the pilot"
else echo "done with $FAILS problem(s): fix them and run again"; fi
exit $FAILS
