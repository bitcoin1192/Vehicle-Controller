import subprocess
from time import sleep

currentDir = "/home/pi/final-skripsi/"
pythonExec = "python3"
def main():
    relayProcess = subprocess.Popen([pythonExec,currentDir+"RelayControl.py"])
    detectorProcess = subprocess.Popen([pythonExec,currentDir+"helmetAnnouncer.py"])
    sleep(3)
    authenticatorProcess = subprocess.Popen([pythonExec,currentDir+"bluetoothVerifier.py"])
    processList = [relayProcess,detectorProcess,authenticatorProcess]
    try:
        watchDogs(processList)
    except KeyboardInterrupt or Exception:
        for process in processList:
            print("Terminating process {}".format(process.pid))
            process.terminate()
        print("All running process is terminated !")
            

def watchDogs(processList):
    while(True):
        for process in processList:
            process.poll()
            print(process.returncode)
            if process.returncode:
                print("Something happen to subprocess")
                raise KeyboardInterrupt
            sleep(2)

main()
