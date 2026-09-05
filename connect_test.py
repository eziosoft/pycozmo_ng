import pycozmo
import time

cli = pycozmo.Client()
cli.start()
cli.connect()

time.sleep(2)
print("Connected to Cozmo successfully")

cli.disconnect()
cli.stop()