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
        print(value)
        if value == '\0':
            self.buf == ""
        else:
            self.buf += value
        
    def ReadValue(self, options):
        return [dbus.Boolean(self.current_status)]

class deviceOwner(gattCharacteristics):
    uniqueOwner = 20
    def ownerUpdate(self, uid):
        self.uniqueOwner = uid
    def ReadValue(self, options):
        return [dbus.Int32(self.uniqueOwner)]
