from pydbus.generic import signal
from pydbus import SessionBus, SystemBus
from gi.repository import GLib
import RPi.GPIO as Pin

busName = "com.sisalma.pydbus.RelayController"
RELAYON = 0
RELAYOFF = 1
RelayPin = 27

class RelayLogic(object):
    """
    <node>
      <interface name='com.sisalma.pydbus.RelayController'>
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
        self.RelayState = RELAYOFF
        self.helmetDetected = False
        self.bluetoothKeyVerified = False

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
            Pin.output(RelayPin,Pin.HIGH)
        elif self.RelayState == RELAYOFF:
            Pin.output(RelayPin,Pin.LOW)
        
    def setupPin():
        Pin.setup(Pin.BOARD)
        Pin.setmode(RelayPin, Pin.OUT)
        
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

    PropertiesChanged = signal()

def main():
    loop = GLib.MainLoop()
    bus = SystemBus()
    bus.publish(busName,RelayLogic())
    loop.run()
    

main()