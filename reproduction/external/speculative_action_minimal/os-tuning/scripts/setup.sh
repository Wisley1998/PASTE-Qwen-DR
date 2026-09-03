#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"

enable_hr_tick() {
    local kernel_config="/boot/config-$(uname -r)"
    echo "[setup] checking HRTICK prerequisites"
    if [[ -r "$kernel_config" ]]; then
      grep -q '^CONFIG_HIGH_RES_TIMERS=y' "$kernel_config" || echo "[setup] warning: CONFIG_HIGH_RES_TIMERS is not enabled in $kernel_config"
      grep -q '^CONFIG_SCHED_HRTICK=y' "$kernel_config" || echo "[setup] warning: CONFIG_SCHED_HRTICK is not enabled in $kernel_config"
    else
      echo "[setup] warning: kernel config $kernel_config is not readable"
    fi

    if [[ -w /sys/kernel/debug/sched/features ]] || sudo -n test -w /sys/kernel/debug/sched/features; then
      echo "[setup] enabling HRTICK in /sys/kernel/debug/sched/features"
      sudo sh -c 'echo HRTICK > /sys/kernel/debug/sched/features'
    else
      echo "[setup] warning: could not enable HRTICK automatically"
    fi
}

echo "[setup] installing system dependencies"
sudo apt-get update
sudo apt-get install -y python3 python3-pip sysbench

if ! mountpoint -q /sys/kernel/debug; then
  echo "[setup] mounting debugfs"
  sudo mount -t debugfs none /sys/kernel/debug || true
fi

echo "[setup] installing Python dependencies"
python3 -m pip install --upgrade pip
python3 -m pip install -r "$ROOT/requirements.txt"
python3 -m pip install -e "$ROOT"

enable_hr_tick

echo "[setup] complete"
