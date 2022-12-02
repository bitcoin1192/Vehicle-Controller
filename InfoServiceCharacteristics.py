import dbus
from bluezdbusInterface.gattCharacteristics import Characteristic as gattCharacteristics
from uuidConstant import *


class lockStatus(gattCharacteristics):
    def __init__(self, bus, index, uuid, flags, service):
        self.current_status = UNLOCKED
        self.buf = ""
        super().__init__(bus, index, uuid, flags, service)

    def statusUpdate(self,lockUpdate):
        self.current_status = lockUpdate

    def WriteValue(self, value, options):
        for byte in value:
            self.buf = self.buf+ chr(byte)
            print(self.buf)
        if self.buf == "unlock":
            self.current_status = UNLOCKED
        elif self.buf == "lock":
            self.current_status = LOCKED
        
    def ReadValue(self, options):
        if self.current_status == UNLOCKED:
            return [dbus.String("Vehicle is Unlocked")]
        elif self.current_status == LOCKED:
            return [dbus.String("Vehicle is Locked")]

class deviceOwner(gattCharacteristics):
    uniqueOwner = 20
    def ownerUpdate(self, uid):
        self.uniqueOwner = uid
    def ReadValue(self, options):
        return [dbus.Int32(self.uniqueOwner)]
