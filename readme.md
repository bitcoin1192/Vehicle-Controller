A python script to lock and unlock vehicle. 
This program depend on BlueZ, PyDBus, Tensorflow, and Adafruit CircuitPython Neopixel Library.

This script takes value sent from Android Apps, via BlueZ, which then stop or start Tensorflow helmet detection model.
The result of helmet detection model is then use to close or open relay connected to Pi Zero GPIO Pin.
