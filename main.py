from AppServices import AppServices
from InfoServiceCharacteristics import deviceOwner, lockStatus
from bluezdbusInterface.gattServices import Service
import uuidConstant

def main():
    #Instantiated apps that hold one or more Service
    apps = AppServices()

    #Instantitated Service that hold characteristics
    deviceStatusService = Service(apps.bus, apps.next_index, uuidConstant.statusCharacteristicsUUID, False)
    
    #Instantiated Characteristics of service
    lockCharacteristic = lockStatus()
    deviceOwnerShip = deviceOwner()
    deviceStatusService.add_characteristic(lockCharacteristic)
    deviceStatusService.add_characteristic(deviceOwnerShip)

    #Register AppServices that hold multiple service
    apps.add_service(deviceStatusService)
    apps.run()
    #apps.quit()