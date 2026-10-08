#!/usr/bin/env bash
#
# Bootstrap PyCozmo-NG: load credentials from .env, make sure the robot is
# reachable, prepare the virtualenv, then run demo.py.
#
# Defaults target the Raspberry Pi deployment (NetworkManager/nmcli joins the
# Cozmo Wi-Fi, apt for install hints), but the script runs anywhere: without
# nmcli it skips the automatic join and just waits for the robot.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

OS="$(uname -s)"
VENV_DIR="$REPO_ROOT/venv"
LOCK_FILE="${TMPDIR:-/tmp}/start.sh.lock"

log()  { printf "\033[1;34m[start.sh]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[start.sh]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[start.sh]\033[0m %s\n" "$*" >&2; }

# --- credentials (.env, gitignored) ----------------------------------------

# Wi-Fi credentials live in .env — the same file demo.py reads via
# python-dotenv. Copy .env.example to .env on each machine.
ENV_FILE="$REPO_ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    err "Missing ${ENV_FILE} — copy .env.example to .env and fill in the Cozmo Wi-Fi credentials."
    exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

COZMO_SSID="${COZMO_WIFI_SSID:-}"
COZMO_PASSWORD="${COZMO_WIFI_PASSWORD:-}"
COZMO_HOST="${COZMO_WIFI_HOST:-172.31.1.1}"

if [[ -z "$COZMO_SSID" || -z "$COZMO_PASSWORD" ]]; then
    err "COZMO_WIFI_SSID / COZMO_WIFI_PASSWORD not set in ${ENV_FILE}."
    exit 1
fi

# --- tooling ----------------------------------------------------------------

command -v ping >/dev/null 2>&1 || { err "ping not found."; exit 1; }

# nmcli (NetworkManager) is only needed for the automatic Wi-Fi join — the
# default Pi setup has it. Elsewhere we fall back to joining manually.
HAVE_NMCLI=0
if command -v nmcli >/dev/null 2>&1; then
    HAVE_NMCLI=1
fi

if ! command -v espeak-ng >/dev/null 2>&1; then
    case "$OS" in
        Darwin) hint="brew install espeak-ng" ;;
        *)      hint="sudo apt-get install espeak-ng" ;;
    esac
    warn "espeak-ng not found — speech output will be unavailable (install: ${hint})."
fi

run_privileged() {
    if command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        "$@"
    fi
}

# Ping timeout flags differ: GNU/Linux take seconds (-W 2), macOS/BSD take
# milliseconds (-W 2000).
ping_ok() {
    case "$OS" in
        Darwin|FreeBSD|OpenBSD|NetBSD) ping -c 1 -W 2000 "$1" >/dev/null 2>&1 ;;
        *)                             ping -c 1 -W 2 "$1" >/dev/null 2>&1 ;;
    esac
}

# --- single instance ---------------------------------------------------------

acquire_lock() {
    if [[ "$OS" == "Linux" ]] && command -v flock >/dev/null 2>&1; then
        exec 9>"$LOCK_FILE"
        if ! flock -n 9; then
            err "Another instance of start.sh is already running. Exiting."
            exit 6
        fi
    else
        if [[ -f "$LOCK_FILE" ]] && kill -0 "$(cat "$LOCK_FILE" 2>/dev/null)" 2>/dev/null; then
            err "Another instance of start.sh is already running (PID $(cat "$LOCK_FILE")). Exiting."
            exit 6
        fi
        echo $$ > "$LOCK_FILE"
    fi
}

# --- robot reachability ------------------------------------------------------

# Wait for the robot to answer ICMP. Prints a hint every 10 s so long waits
# are not silent.
wait_for_robot() {
    local max_wait="$1"
    local elapsed=0
    while (( elapsed < max_wait )); do
        if ping_ok "$COZMO_HOST"; then
            return 0
        fi
        if (( elapsed > 0 && elapsed % 10 == 0 )); then
            log "Still waiting for Cozmo at ${COZMO_HOST} (${elapsed}/${max_wait} s)..."
        fi
        sleep 1
        (( elapsed += 1 ))
    done
    return 1
}

wait_for_wifi() {
    local elapsed=0
    local max_wait=30
    while (( elapsed < max_wait )); do
        if nmcli -t -f ACTIVE,SSID dev wifi 2>/dev/null | grep -qi "^yes:${COZMO_SSID}$"; then
            return 0
        fi
        sleep 1
        (( elapsed += 1 ))
    done
    return 1
}

# The Wi-Fi interface is usually wlan0 on the Pi; discover it when possible.
wifi_iface() {
    nmcli -t -f DEVICE,TYPE device status 2>/dev/null | awk -F: '$2 == "wifi" { print $1; exit }'
}

ensure_cozmo_wifi() {
    if ping_ok "$COZMO_HOST"; then
        log "Cozmo reachable at ${COZMO_HOST}."
        return 0
    fi

    # No NetworkManager (macOS, minimal containers, ...): the user joins the
    # network themselves — give them time and a clear message.
    if [[ "$HAVE_NMCLI" -eq 0 ]]; then
        warn "nmcli not found — skipping the automatic Wi-Fi join."
        warn "Join the '${COZMO_SSID}' network manually if needed; waiting up to 30 s for the robot..."
        if wait_for_robot 30; then
            log "Connected to Cozmo (${COZMO_HOST})."
            return 0
        fi
        err "Cozmo at ${COZMO_HOST} is not reachable."
        err "Power the robot on and put this machine on its '${COZMO_SSID}' network (set in .env)."
        exit 2
    fi

    # NetworkManager path (default on the Pi).
    if nmcli -t -f ACTIVE,SSID dev wifi 2>/dev/null | grep -qi "^yes:${COZMO_SSID}$"; then
        err "Associated with ${COZMO_SSID} but ${COZMO_HOST} is not reachable."
        err "Is the robot powered on?"
        exit 3
    fi

    local iface
    iface="$(wifi_iface)"
    log "Bringing up Wi-Fi..."
    nmcli radio wifi on >/dev/null 2>&1 || true
    run_privileged nmcli dev set "${iface:-wlan0}" managed yes >/dev/null 2>&1 || true

    if ! nmcli -t -f NAME connection show | grep -qx "${COZMO_SSID}" 2>/dev/null; then
        log "Saving ${COZMO_SSID} profile..."
        run_privileged nmcli dev wifi connect "${COZMO_SSID}" password "${COZMO_PASSWORD}" \
            802-11-wireless.cloned-mac-address permanent \
            connection.autoconnect yes \
            connection.permissions user:"$(id -un)" \
            ipv4.never-default yes \
            ipv6.never-default yes || true
    fi

    log "Activating ${COZMO_SSID}..."
    run_privileged nmcli connection up "${COZMO_SSID}" >/dev/null 2>&1 || true

    if ! wait_for_wifi; then
        err "Failed to join ${COZMO_SSID}."
        err "Is the robot powered on and its AP visible? (SSID/password come from .env)"
        exit 2
    fi

    if ! wait_for_robot 15; then
        err "Joined ${COZMO_SSID} but ${COZMO_HOST} is not reachable."
        err "Is the robot powered on?"
        exit 3
    fi
    log "Connected to Cozmo (${COZMO_HOST})."
}

# --- virtualenv ----------------------------------------------------------------

ensure_venv() {
    if [[ -d "$VENV_DIR" ]]; then
        return 0
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        err "python3 not found — install Python 3.10+ first."
        exit 5
    fi
    log "No virtualenv at ${VENV_DIR} — creating it..."
    if ! python3 -m venv "$VENV_DIR"; then
        err "Failed to create the virtualenv (on Debian/Ubuntu: sudo apt-get install python3-venv)."
        exit 5
    fi
    log "Installing PyCozmo-NG (first run only, may take a while)..."
    if ! "${VENV_DIR}/bin/pip" install -e "$REPO_ROOT"; then
        err "pip install -e . failed — see the output above."
        exit 5
    fi
}

# --- main -------------------------------------------------------------------------

acquire_lock

ensure_cozmo_wifi

ensure_venv

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "Python: $(python --version 2>&1) (venv=${VIRTUAL_ENV:-inactive})"
log "Launching demo.py (Ctrl-C to stop)..."
exec python demo.py "$@"
