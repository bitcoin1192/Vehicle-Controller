from pydbus.generic import signal as dbus_signal
from pydbus import SessionBus, SystemBus
from gi.repository import GLib
import RPi.GPIO as Pin
import time
import math
from uuidConstant import LOCKED, UNLOCKED, TEST, FACEFOUND, FACENOTFOUND, HELMETFOUND
try:
    import board
    import neopixel
    RelayPin = 22
    LEDPIN = board.D18
    bus = SystemBus()
    runOnPI = True
    print("Running on Pi: \n Relay Pin: {}, LED Pin: {}".format(RelayPin,LEDPIN))
except ModuleNotFoundError:
    #Assuming not running on pi
    RelayPin = 22
    LEDPIN = 18
    bus = SessionBus()
    runOnPI = False
    print("Not on a Pi")


busName = "com.sisalma.pydbus"
RELAYON = 0
RELAYOFF = 1

class RelayLogic:
    """
    <node>
      <interface name='com.sisalma.pydbus'>
        <method name='BluetoothKeyStatus'>
          <arg type='n' name='a' direction='in'/>
          <arg type='b' name='response' direction='out'/>
        </method>
        <method name='HelmetStatus'>
          <arg type='n' name='a' direction='in'/>
          <arg type='b' name='response' direction='out'/>
        </method>
        <property name="bluetoothKeyVerified" type="n" access="read">
          <annotation name="org.freedesktop.DBus.Property.EmitsChangedSignal" value="true"/>
        </property>
      </interface>
    </node>
    """
    def __init__(self):
        self.useLED = runOnPI
        if self.useLED:
            self.pixels = neopixel.NeoPixel(board.D12, 1)
        self.RelayState = RELAYOFF
        self._helmetDetected = False
        self._bluetoothKeyVerified = LOCKED
        Pin.setmode(Pin.BCM)
        Pin.setup(RelayPin, Pin.OUT)
        self.lockUnlockIO()
        

    def BluetoothKeyStatus(self, s):
        #Only trigger on State Changes
        #if s != self._bluetoothKeyVerified:
        self._bluetoothKeyVerified = s
        self.lockUnlockIO()
        
    def HelmetStatus(self, s):
        #Only trigger on State Changes
        #if s != self._helmetDetected:
        self._helmetDetected = s
        self.lockUnlockIO()
    
    def lockUnlockIO(self):
        if self._bluetoothKeyVerified == UNLOCKED and self._helmetDetected == HELMETFOUND:
            self.RelayState = RELAYON
            #self.changeLEDColor(LEDGREEN)
        elif self._bluetoothKeyVerified == UNLOCKED and self._helmetDetected == FACEFOUND:
            #self.changeLEDColor(LEDYELLOW)
            self.RelayState = RELAYOFF
        elif self._bluetoothKeyVerified == UNLOCKED and self._helmetDetected == FACENOTFOUND:
            #self.changeLEDColor(LEDBLUE)
            self.RelayState = RELAYOFF
        elif self._bluetoothKeyVerified == LOCKED :
            self.RelayState = RELAYOFF
            #self.changeLEDColor(LEDRED)
            #Send signal to stop detecting helmet on other script listening
            self.PropertiesChanged(busName, {"bluetoothKeyVerified": self._bluetoothKeyVerified}, [])
        self.stateAction()
        
    def stateAction(self):
        if self.RelayState == RELAYON:
            Pin.output(RelayPin,Pin.HIGH)
            #print("Relay state changes to: ON")
        elif self.RelayState == RELAYOFF:
            Pin.output(RelayPin,Pin.LOW)
            #print("Relay state changes to: OFF")
        
        
    @property
    def bluetoothKeyVerified(self):
        return self._bluetoothKeyVerified

    def changeLEDColor(self,color):
        if self.useLED:
            self.pixels[0] = color
        else:
            print("Program not running on Pi, disabling neopixel library")
    
    def stopObject(self):
        pass
        #self.changeLEDColor(LEDOFF) 

    PropertiesChanged = dbus_signal()

def main():
    loop = GLib.MainLoop()
    RelayObject = RelayLogic()
    bus.publish(busName,RelayObject)
    loop.run()
    
main()
