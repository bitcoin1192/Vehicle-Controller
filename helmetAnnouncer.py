from pydbus import SessionBus, SystemBus
from camera import detector
from cv2 import VideoCapture
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
    vidcap = VideoCapture(0)
    time.sleep(5000)
    if vidcap.isOpened() is True:
        td = detector.ImageProcessor(VideoCapture(0))
        fd = detector.FaceDetector(td,128,128,1.7)
        hd = detector.HelmetDetector(fd,0.7,4)
        Controller = bus.get("com.sisalma.pydbus")
        try:
            while(True):
                if Controller.bluetoothKeyVerified == True:
                    hd.stopFlags(False)
                else:
                    print("Pausing helmet detection: BluetoothKey is not verified")
                    hd.stopFlags(True)
                    hd.resetCounter()
                    time.sleep(1)
                result = hd.tallyResult()
                print(result)
                Controller.HelmetStatus(result)
        except GLib.Error as err:
            print("Complementary program exitting")
    else:
        print("Camera not found or timeout")
        raise Exception("Camera not found or timeout")
main()
