#!/usr/bin/env bash
# Prints what a jetson has: model, memory, software, disks, network. Changes nothing.
echo "=== $(hostname)"
echo "address:  $(hostname -I | awk '{print $1}')"
echo "model:    $(tr -d '\0' < /proc/device-tree/model)"
echo "memory:   $(free -g | awk '/Mem:/{print $2" GB"}')"
echo "jetpack:  $(head -1 /etc/nv_tegra_release | sed -E 's/# R([0-9]+) \(release\), REVISION: ([0-9.]+).*/L4T \1.\2/')"
echo "deepstream: $(grep -h Version /opt/nvidia/deepstream/deepstream/version 2>/dev/null || echo 'not installed')"
echo "docker:   $(docker --version 2>/dev/null || echo 'not installed')"
echo "disks:"; lsblk -d -o NAME,SIZE,TYPE | grep -vE "loop|zram|boot"
echo "free on /: $(df -h / | awk 'NR==2{print $4}')"
echo "can see the pilot: $(ping -c1 -W2 192.168.0.139 >/dev/null && echo yes || echo no)"
echo "VLM server here: $(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8010/v1/models 2>/dev/null || echo none)"
