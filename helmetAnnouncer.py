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
    if vidcap.isOpened() is True:
        td = detector.ImageProcessor(vidcap)
        fd = detector.FaceDetector(td,128,128,1.7)
        hd = detector.HelmetDetector(fd,0.75,2)
        Controller = bus.get("com.sisalma.pydbus")
        try:
            while(True):
                if Controller.bluetoothKeyVerified == False:
                    hd.stopFlags(True)
                    time.sleep(0.5)
                else:
                    #print("Pausing helmet detection: BluetoothKey is not verified")
                    hd.stopFlags(False)
                    hd.resetCounter()
                result = hd.tallyResult()
                if result == None:
                    print("Detection not started")
                else:
                    print(result)
                    Controller.HelmetStatus(result)
        except GLib.Error as err:
            print("Complementary program exitting")
    else:
        print("Camera not found or timeout")
main()
