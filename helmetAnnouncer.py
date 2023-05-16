from pydbus import SessionBus, SystemBus
from camera import detector
from cv2 import VideoCapture
from gi.repository import GLib
from uuidConstant import LOCKED, UNLOCKED, TEST
import time
try:
    import board
    import neopixel
    bus = SystemBus()
    print("Warning: Using system bus")
except ModuleNotFoundError:
    bus = SessionBus()
    print("Warning: Using session bus")


def main():
    vidcap = VideoCapture(0)
    if vidcap.isOpened() is True:
        td = detector.ImageProcessor(vidcap)
        fd = detector.FaceDetector(td,128,128,1.7)
        hd = detector.HelmetDetector(fd,0.65,2)
        Controller = bus.get("com.sisalma.pydbus")
        try:
            while(True):
                if Controller.bluetoothKeyVerified == LOCKED:
                    print("Pausing detection!")
                    hd.stopFlags(True)
                    hd.resetCounter()
                    time.sleep(0.35)
                elif Controller.bluetoothKeyVerified == TEST:
                    hd.stopFlags(False)
                    print("Bluetooth: TEST")
                elif Controller.bluetoothKeyVerified == UNLOCKED:
                    hd.stopFlags(False)
                result = hd.tallyResult()
                if result == None:
                    print("Warning: Detection not started")
                else:
                    print("Result is: {}".format(result))
                    Controller.HelmetStatus(result)
        except GLib.Error as err:
            print("Error: Complementary program exitting")
    else:
        print("Error: Camera not found or timeout")
main()
