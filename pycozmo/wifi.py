import os
import socket
import subprocess
import time

from pycozmo.logger import logger


def get_local_ip() -> str:
    """Best-effort local IPv4 address other machines use to reach this host.

    Uses the kernel's UDP routing table (no packet is actually sent) to find
    the address bound on the default route — e.g. the LAN address on the Pi
    (eth0), or the Cozmo AP address when connected directly.

    Returns:
        Dotted-quad IPv4 address, or ``127.0.0.1`` if it cannot be determined.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))  # no traffic sent — UDP connect only
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def connect_to_cozmo_wifi():
    """Auto-connect to Cozmo WiFi AP if not already connected."""
    ssid = os.environ.get("COZMO_WIFI_SSID")
    password = os.environ.get("COZMO_WIFI_PASSWORD")
    if not ssid:
        logger.info("No COZMO_WIFI_SSID configured - skipping Wi-Fi AP check")
        return True

    # Check if already connected to this SSID via nmcli
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "NAME,DEVICE,STATE", "con", "show"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[0] == ssid and parts[2] == "activated":
                logger.info("Already connected to Wi-Fi AP %s", ssid)
                return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Connect via nmcli
    logger.info("Connecting to Wi-Fi AP %s...", ssid)
    try:
        result = subprocess.run(
            ["sudo", "nmcli", "dev", "wifi", "connect", ssid, "password", password],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            logger.info("Connected to Wi-Fi AP %s (waiting 2 s for DHCP)", ssid)
            time.sleep(2)  # Wait for DHCP
            return True
        else:
            logger.warning(
                "Wi-Fi connect to %s failed: %s", ssid, result.stderr.strip()
            )
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("Wi-Fi connect to %s error: %s", ssid, e)
        return False
