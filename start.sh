#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

COZMO_SSID="Cozmo_7144B4"
COZMO_PASSWORD="973-37134-7253402"
COZMO_HOST="172.31.1.1"
VENV_DIR="$REPO_ROOT/venv"

log() { printf "\033[1;34m[start.sh]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[start.sh]\033[0m %s\n" "$*" >&2; }

if [[ "$(uname -s)" != "Linux" ]]; then
    err "This script must run on the Raspberry Pi (detected: $(uname -s))."
    err "Sync to the Pi first: ./sync_to_pi.sh"
    exit 1
fi

command -v nmcli >/dev/null 2>&1 || { err "nmcli not found. Install NetworkManager."; exit 1; }
command -v ping   >/dev/null 2>&1 || { err "ping not found."; exit 1; }

ensure_cozmo_wifi() {
    if nmcli -t -f ACTIVE,SSID dev wifi 2>/dev/null | grep -qi "^yes:${COZMO_SSID}$"; then
        log "Already connected to ${COZMO_SSID}."
        return 0
    fi

    if ping -c 1 -W 2 "$COZMO_HOST" >/dev/null 2>&1; then
        log "Cozmo ${COZMO_HOST} reachable."
        return 0
    fi

    log "Bringing up wlan0..."
    nmcli radio wifi on >/dev/null 2>&1 || true
    sudo nmcli dev set wlan0 managed yes >/dev/null 2>&1 || true

    if ! nmcli -t -f NAME connection show | grep -qx "${COZMO_SSID}"; then
        log "Saving ${COZMO_SSID} profile..."
        sudo nmcli dev wifi connect "${COZMO_SSID}" password "${COZMO_PASSWORD}" \
            802-11-wireless.cloned-mac-address permanent \
            connection.autoconnect yes \
            connection.permissions user:"$(id -un)" \
            ipv4.never-default yes \
            ipv6.never-default yes
    else
        log "Activating ${COZMO_SSID}..."
        sudo nmcli connection up "${COZMO_SSID}"
    fi

    if ! nmcli -t -f ACTIVE,SSID dev wifi 2>/dev/null | grep -qi "^yes:${COZMO_SSID}$"; then
        err "Failed to associate with ${COZMO_SSID}."
        err "Is Cozmo powered on and its AP visible?"
        exit 2
    fi

    if ! ping -c 1 -W 5 "$COZMO_HOST" >/dev/null 2>&1; then
        err "Connected to ${COZMO_SSID} but ${COZMO_HOST} not reachable."
        err "Is Cozmo on and on the same AP?"
        exit 3
    fi

    log "Connected to Cozmo (${COZMO_HOST})."
}

if ! ping -c 1 -W 2 192.168.10.1 >/dev/null 2>&1; then
    err "Wired management link (192.168.10.0/24) is down. Aborting."
    exit 4
fi

ensure_cozmo_wifi

if [[ ! -d "$VENV_DIR" ]]; then
    err "Virtualenv not found at ${VENV_DIR}."
    err "Create it: python3 -m venv \"${VENV_DIR}\" && \"${VENV_DIR}/bin/pip\" install -e ."
    exit 5
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "Python: $(python --version 2>&1) (venv=${VIRTUAL_ENV:-inactive})"
log "Launching main.py (Ctrl-C to stop)..."
exec python main.py "$@"