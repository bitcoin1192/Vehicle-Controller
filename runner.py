import subprocess
from time import sleep

currentDir = "/home/pi/final-skripsi/"
def main():
    relayProcess = subprocess.Popen(currentDir+"RelayControl.py")
    detectorProcess = subprocess.Popen(currentDir+"helmetAnnouncer.py")
    authenticatorProcess = subprocess.Popen(currentDir+"bluetoothVerifier.py")
    processList = [relayProcess,detectorProcess,authenticatorProcess]
    try:
        watchDogs(processList)
    except KeyboardInterrupt:
        for process in processList:
            print("Terminating process {}".format(process.pid))
            process.terminate()
        print("All running process is terminated !")
            

def watchDogs(processList):
    while(True):
        for process in processList:
            if process.stderr:
                raise KeyboardInterrupt
        sleep(2)