import urllib.request
import time
import subprocess
import os
import signal

p = subprocess.Popen(["docker", "run", "--rm", "-p", "8000:8000", "supply-chain-sim"])
time.sleep(5)
try:
    with urllib.request.urlopen("http://localhost:8000/health") as response:
        print(response.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
finally:
    p.send_signal(signal.SIGINT)
    p.wait()
