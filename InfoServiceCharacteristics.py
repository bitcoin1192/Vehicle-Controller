import dbus
from bluezdbusInterface.gattCharacteristics import Characteristic as gattCharacteristics
from uuidConstant import *
from bluezdbusInterface.interfaceConstant import *
from pydbus import SystemBus


class lockStatus(gattCharacteristics):
    def __init__(self, bus, index, uuid, flags, service):
        self.current_status = LOCKED
        self.buf = ""
        self.otherbus = SystemBus()
        self.relayControl = self.otherbus.get("com.sisalma.pydbus")
        super().__init__(bus, index, uuid, flags, service)

    def statusUpdate(self,lockUpdate):
        self.current_status = lockUpdate

    def WriteValue(self, value, options):
        for byte in value:
            self.buf = self.buf + chr(byte)
            print(self.buf)
        if self.buf == "unlock" or self.buf == "u":
            print("Unlocking")
            self.relayControl.BluetoothKeyStatus(True)
            self.current_status = UNLOCKED
        elif self.buf == "lock" or self.buf == "a":
            print("Locking")
            self.relayControl.BluetoothKeyStatus(False)
            self.current_status = LOCKED
        self.buf = ""

    @dbus.service.method(GATT_CHRC_IFACE,
                        in_signature='a{sv}',
                        out_signature='ay')    
    def ReadValue(self, options):
        temp = ""
        msg = []
        print("Request")
        if self.current_status == UNLOCKED:
           temp = "Vehicle is Unlocked"
        elif self.current_status == LOCKED:
           temp = "Vehicle is Locked"
        for char in temp:
            msg.append(ord(char))
        return msg

class deviceOwner(gattCharacteristics):
    uniqueOwner = 20
    def ownerUpdate(self, uid):
        self.uniqueOwner = uid
    def ReadValue(self, options):
        return [dbus.Int32(self.uniqueOwner)]
