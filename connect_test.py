import pycozmo
import time

cli = pycozmo.Client()
cli.start()
cli.connect()

time.sleep(2)
try:
    ver = cli.conn.version
    print("Connected! Firmware version:", ver)
except Exception:
    print("Connected but no version info")

packet_count = cli.conn.pkt_cnt
print("Packets received:", packet_count)

cli.disconnect()
cli.stop()