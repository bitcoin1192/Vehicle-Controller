from pydbus.generic import signal
from pydbus import SessionBus, SystemBus
from gi.repository import GLib
import RPi.GPIO as Pin
import time
import board
import neopixel
import math

busName = "com.sisalma.pydbus"
RELAYON = 0
RELAYOFF = 1
RelayPin = 22
LEDGREEN = (0,255,12)
LEDRED = (255,12,0)
LEDOFF = (0,0,0)
LEDPIN = board.D18

class RelayLogic:
    """
    <node>
      <interface name='com.sisalma.pydbus'>
        <method name='BluetoothKeyStatus'>
          <arg type='b' name='a' direction='in'/>
          <arg type='b' name='response' direction='out'/>
        </method>
        <method name='HelmetStatus'>
          <arg type='b' name='a' direction='in'/>
          <arg type='b' name='response' direction='out'/>
        </method>
        <property name="LockStatus" type="s" access="readwrite">
          <annotation name="org.freedesktop.DBus.Property.EmitsChangedSignal" value="true"/>
        </property>
      </interface>
    </node>
    """
    def __init__(self):
        self.pixels = neopixel.NeoPixel(board.D18, 1)
        self.RelayState = RELAYOFF
        self.helmetDetected = False
        self.bluetoothKeyVerified = False
        self.setupPin()

    def BluetoothKeyStatus(self, s):
        self.bluetoothKeyVerified = s
        self.lockUnlockIO()
        
    def HelmetStatus(self, s):
        self.helmetDetected = s
        self.lockUnlockIO()
    
    def lockUnlockIO(self):
        if self.bluetoothKeyVerified and self.helmetDetected:
            self.RelayState = RELAYON
        else:
            self.RelayState = RELAYOFF
        self.stateAction()
        
    def stateAction(self):
        if self.RelayState == RELAYON:
            self.pixels[0] = LEDGREEN
            Pin.output(RelayPin,Pin.HIGH)
        elif self.RelayState == RELAYOFF:
            self.pixels[0] = LEDRED
            Pin.output(RelayPin,Pin.LOW)
        
    def setupPin(self):
        Pin.setmode(Pin.BCM)
        Pin.setup(RelayPin, Pin.OUT)
        self.stateAction()
        
    @property
    def LockStatus(self):
        if self.RelayState == RELAYON:
            return "Electrical System is ON!"
        elif self.RelayState == RELAYOFF:
            return "Off"

    @LockStatus.setter
    def LockStatus(self, value):
        self._someProperty = value
        self.PropertiesChanged(busName, {"LockStatus": self.LockStatus}, [])

    def Stop(self):
        self.pixels[0] = LEDOFF

    PropertiesChanged = signal()

fun ToggleLED
def main():
    loop = GLib.MainLoop()
    #bus = SessionBus()
    bus = SystemBus()
    RelayObject = RelayLogic()
    bus.publish(busName,RelayObject)
    loop.run()
    
main()