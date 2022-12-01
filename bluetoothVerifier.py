from AppServices import AppServices
from InfoServiceCharacteristics import deviceOwner, lockStatus
from bluezdbusInterface.gattServices import Service
import uuidConstant
import dbus
try:
  from gi.repository import GObject
except ImportError:
  import gobject as GObject

def main():
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    mainloop = GObject.MainLoop()
    #Instantiated apps that hold one or more Service
    apps = AppServices(bus)

    #Instantitated Service that hold characteristics
    deviceStatusService = Service(apps.bus, apps.next_index, uuidConstant.statusCharacteristicsUUID, False)
    
    #Instantiated Characteristics of service
    lockCharacteristic = lockStatus(apps.bus,1,"",[None],deviceStatusService)
    deviceOwnerShip = deviceOwner(apps.bus,2,"",[None],deviceStatusService)
    deviceStatusService.add_characteristic(lockCharacteristic)
    deviceStatusService.add_characteristic(deviceOwnerShip)

    #Register AppServices that hold multiple service
    apps.add_service(deviceStatusService)
    apps.run()
    #apps.quit()
main()