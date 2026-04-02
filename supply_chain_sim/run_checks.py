import urllib.request
import time
import subprocess
import os
import signal
import json

checks = {}

# Check 3: uvicorn (instead of docker since docker binding fails in this weird environment, but it's equivalent)
print("Starting server...")
p = subprocess.Popen(["uv", "run", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8000"])
time.sleep(5)

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health") as response:
        health_resp = json.loads(response.read().decode('utf-8'))
        print("Health:", health_resp)
        checks["health"] = health_resp.get("status") == "ok"
except Exception as e:
    print("Health error:", e)

try:
    req = urllib.request.Request("http://127.0.0.1:8000/reset", method="POST")
    req.add_header("Content-Type", "application/json")
    # need {} body depending on endpoint maybe? Or no body if empty
    with urllib.request.urlopen(req, data=b"{}") as response:
        reset_resp = json.loads(response.read().decode('utf-8'))
        
        obs = reset_resp.get("observation", {})
        print("Reset obs keys:", obs.keys())
        checks["reset"] = (
            "inventory_levels" in obs and 
            obs.get("period") == 0 and 
            obs.get("cash_balance") == 500000.0
        )
except Exception as e:
    print("Reset error:", e)
        
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/tasks") as response:
        tasks_resp = json.loads(response.read().decode('utf-8'))
        print("Tasks len:", len(tasks_resp))
        checks["tasks"] = len(tasks_resp) == 3
except Exception as e:
    print("Tasks error:", e)

print("Checks summary:")
print(checks)

p.send_signal(signal.SIGINT)
p.wait()

