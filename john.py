import control
import signal
import sys

ctrl = control.Control(sys.argv[1])

ctrl.stand_up()
ctrl.move(0.25, 0, 0.02)

def signal_handler(signal, frame):
  print('You pressed Ctrl+C!')
  ctrl.stop()
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.pause()