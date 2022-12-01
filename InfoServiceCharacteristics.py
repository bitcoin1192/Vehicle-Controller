import dbus
from bluezdbusInterface.gattCharacteristics import Characteristic as gattCharacteristics
from uuidConstant import *


class lockStatus(gattCharacteristics):
    current_status = UNLOCKED
    def statusUpdate(self,lockUpdate):
        self.current_status = lockUpdate

    def ReadValue(self, options):
        return [dbus.Boolean(self.current_status)]

class deviceOwner(gattCharacteristics):
    uniqueOwner = 20
    def ownerUpdate(self, uid):
        self.uniqueOwner = uid
    def ReadValue(self, options):
        return [dbus.Int32(self.uniqueOwner)]