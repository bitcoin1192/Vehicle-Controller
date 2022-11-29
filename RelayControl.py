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
        <property name="bluetoothKeyVerified" type="b" access="read">
          <annotation name="org.freedesktop.DBus.Property.EmitsChangedSignal" value="true"/>
        </property>
      </interface>
    </node>
    """
    def __init__(self):
        self.pixels = neopixel.NeoPixel(board.D18, 1)
        self.RelayState = RELAYOFF
        self._helmetDetected = False
        self._bluetoothKeyVerified = False
        Pin.setmode(Pin.BCM)
        Pin.setup(RelayPin, Pin.OUT)
        self.stateAction()

    def BluetoothKeyStatus(self, s):
        self._bluetoothKeyVerified = s
        self.lockUnlockIO()
        
    def HelmetStatus(self, s):
        self._helmetDetected = s
        self.lockUnlockIO()
    
    def lockUnlockIO(self):
        if self._bluetoothKeyVerified and self._helmetDetected:
            self.RelayState = RELAYON
        elif self._bluetoothKeyVerified and not self._helmetDetected:
            self.RelayState = RELAYOFF
        elif not self._bluetoothKeyVerified :
            #Send signal to stop detecting helmet on other script listening
            pass
        self.stateAction()
        
    def stateAction(self):
        if self.RelayState == RELAYON:
            self.pixels[0] = LEDGREEN
            Pin.output(RelayPin,Pin.HIGH)
        elif self.RelayState == RELAYOFF:
            self.pixels[0] = LEDRED
            Pin.output(RelayPin,Pin.LOW)
        
    @property
    def _BluetoothKeyVerified(self):
        return self._bluetoothKeyVerified

    @_BluetoothKeyVerified.setter
    def LockStatus(self, value):
        self._someProperty = value
        self.PropertiesChanged(busName, {"LockStatus": self.LockStatus}, [])

    def Stop(self):
        self.pixels[0] = LEDOFF

    PropertiesChanged = signal()

def main():
    loop = GLib.MainLoop()
    #bus = SessionBus()
    bus = SystemBus()
    RelayObject = RelayLogic()
    bus.publish(busName,RelayObject)
    loop.run()
    
main()