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
    hd = detector.HelmetDetector(fd,0.7)
    Controller = bus.get("com.sisalma.pydbus")
    try:
        while(True):
            if Controller.bluetoothKeyVerified:
                result = hd.helmetUsed(12)
                print(result)
                Controller.HelmetStatus(result)
            else:
                print("Won't run: BluetoothKey is not verified")
                time.sleep(1)
    except GLib.Error as err:
        print("Complementary program exitting")
main()
