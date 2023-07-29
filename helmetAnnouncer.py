from pydbus import SessionBus, SystemBus
from camera import detector
from cv2 import VideoCapture
from gi.repository import GLib
from uuidConstant import LOCKED, UNLOCKED, TEST, FACEFOUND, FACENOTFOUND, HELMETFOUND
import time
try:
    import board
    import neopixel
    pixels = neopixel.NeoPixel(board.D12, 1)
    useLED = True
    RelayPin = 22
    LEDPIN = board.D18
    bus = SystemBus()
    print("Warning: Using system bus")
except ModuleNotFoundError:
    bus = SessionBus()
    print("Warning: Using session bus")

LEDGREEN = (0,255,12)
LEDYELLOW = (255,255,0)
LEDBLUE = (12,0,255)
LEDRED = (255,12,0)
LEDOFF = (0,0,0)

def changeLEDColor(color):
        if useLED:
            pixels[0] = color
        else:
            print("Program not running on Pi, disabling neopixel library")
def main():
    vidcap = VideoCapture(0)
    #vidcap = VideoCapture("outdoor-helm-ir-ariq.mp4")
    if vidcap.isOpened() is True:
        td = detector.ImageProcessor(vidcap)
        fd = detector.FaceDetector(td,128,128,1.7)
        hd = detector.HelmetDetector(fd,0.65,2)
        Controller = bus.get("com.sisalma.pydbus")
        try:
            while(True):
                if Controller.bluetoothKeyVerified == LOCKED:
                    #print("Pausing detection!")
                    changeLEDColor(LEDRED)
                    hd.stopFlags(True)
                    hd.resetCounter()
                    time.sleep(0.05)
                elif Controller.bluetoothKeyVerified == TEST:
                    hd.stopFlags(False)
                    print("Bluetooth: TEST")
                elif Controller.bluetoothKeyVerified == UNLOCKED:
                    hd.stopFlags(False)
                result = hd.tallyResult()
                if result == None:
                    Controller.HelmetStatus(FACENOTFOUND)
                elif result == FACENOTFOUND:
                    changeLEDColor(LEDBLUE) 
                    Controller.HelmetStatus(FACENOTFOUND)
                    #print("Warning: Detection not started")
                elif result == FACEFOUND:
                    changeLEDColor(LEDYELLOW)
                    Controller.HelmetStatus(result)
                elif result == HELMETFOUND:
                    changeLEDColor(LEDGREEN)
                    Controller.HelmetStatus(result)
                        
                    
        except GLib.Error as err:
            print("Error: Complementary program exitting")
    else:
        print("Error: Camera not found or timeout")
main()
