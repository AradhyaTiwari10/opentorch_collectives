#!/bin/bash
set -e
echo "Starting FastAPI server for testing endpoints..."
uv run uvicorn server.app:app --host 127.0.0.1 --port 8000 > /dev/null 2>&1 &
SERVER_PID=$!
sleep 3

# Check 3: health
HEALTH=$(curl -s http://127.0.0.1:8000/health)
echo "Health payload: $HEALTH"

# Check 4: reset
RESET=$(curl -s -X POST http://127.0.0.1:8000/reset -H "Content-Type: application/json" -d '{"seed": 42}')
echo "Reset payload length: ${#RESET}"
echo "$RESET" | grep -o '"period":0' || echo "Missing period:0"
echo "$RESET" | grep -o '"cash_balance":500000.0' || echo "Missing cash_balance:500000.0"

# Check 5: tasks
TASKS=$(curl -s http://127.0.0.1:8000/tasks)
echo "Tasks length: $(echo $TASKS | python -c 'import sys, json; print(len(json.load(sys.stdin)))')"

# Cleanup
kill $SERVER_PID
echo "Done"
