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
        self.notifyReq = False
        super().__init__(bus, index, uuid, flags, service)

    def statusUpdate(self,lockUpdate):
        self.current_status = lockUpdate
    
    @dbus.service.method(GATT_CHRC_IFACE,
                        in_signature='aya{sv}')
                        #,out_signature='ay')
    def WriteValue(self, value, options):
        for byte in value:
            self.buf = self.buf + chr(byte)
            print(self.buf)
        if self.buf == "unlock" or self.buf == "u":
            print("Unlocking")
            self.relayControl.BluetoothKeyStatus(UNLOCKED)
            self.current_status = UNLOCKED
        elif self.buf == "lock" or self.buf == "a":
            print("Locking")
            self.relayControl.BluetoothKeyStatus(LOCKED)
            self.current_status = LOCKED
        elif self.buf[0:3] == "test":
            self.relayControl.BluetoothKeyStatus(TEST)
        self.buf = ""
        retMessage = []
        for char in "notiftest":
            retMessage.append(ord(char))
        if self.notifyReq:
            self.PropertiesChanged(GATT_CHRC_IFACE,{'Value': retMessage},[])

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
    
    def StartNotify(self):
        self.notifyReq = True
        
    def StopNotify(self):
        self.notifyReq = False

class deviceOwner(gattCharacteristics):
    uniqueOwner = 20
    def ownerUpdate(self, uid):
        self.uniqueOwner = uid
    def ReadValue(self, options):
        return [dbus.Int32(self.uniqueOwner)]

class deviceInfo(gattCharacteristics):
    @dbus.service.method(GATT_CHRC_IFACE,
                        in_signature='a{sv}',
                        out_signature='ay')
    def ReadValue(self, options):
        msg = []
        for char in "Honda Beat":
             msg.append(ord(char))
        return msg
