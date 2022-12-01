from __future__ import print_function
import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
from bluezdbusInterface.interfaceConstant import *


def find_adapter(bus, adapter_interface_name, adapter_name):
    remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/'), DBUS_OM_IFACE)
    objects = remote_om.GetManagedObjects()

    for o, props in objects.items():
        print('checking adapter %s, keys: %s' % (o, props.keys()))
        if adapter_interface_name in props.keys():
            print('found adapter %s' % (o,))
            if '/' + adapter_name in o:
                print('returning adapter %s' % (o,))
                return o

    return None