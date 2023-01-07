from pydbus import SessionBus, SystemBus
from camera import detector
from cv2 import VideoCapture as selectCamera
from gi.repository import GLib
import time
try:
    import board
    import neopixel
    bus = SystemBus()
    print("Using system bus")
except ModuleNotFoundError:
    bus = SessionBus()
    print("Using session bus")


def main():
    td = detector.ImageProcessor(selectCamera(0))
    fd = detector.FaceDetector(td,128,128,1.7)
    hd = detector.HelmetDetector(fd,0.7,6)
    Controller = bus.get("com.sisalma.pydbus")
    try:
        while(True):
            if Controller.bluetoothKeyVerified == True:
                result = hd.tallyResult()
                print(result)
                Controller.HelmetStatus(result)
            else:
                print("Pausing helmet detection: BluetoothKey is not verified")
                hd.stopFlags()
                time.sleep(1)
    except GLib.Error as err:
        print("Complementary program exitting")
main()
