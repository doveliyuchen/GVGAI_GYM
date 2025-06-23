#!/bin/bash
echo "Stopping GVGAI experiments..."

# Stop Python processes
echo "Stopping Python processes..."
pkill -f "main_4.24.py"
pkill -f "portkey"

# Stop Java GVGAI servers
echo "Stopping GVGAI Java servers..."
pkill -f "GVGAI"

# Force kill if needed
sleep 2
pkill -9 -f "main_4.24.py" 2>/dev/null
pkill -9 -f "GVGAI" 2>/dev/null

echo "All GVGAI experiment processes stopped!"

# Verify
echo "Checking remaining processes..."
remaining=$(ps aux | grep -E "(main_4.24|GVGAI)" | grep -v grep | wc -l)
if [ $remaining -eq 0 ]; then
    echo "✅ All processes successfully stopped!"
else
    echo "⚠️  Some processes may still be running:"
    ps aux | grep -E "(main_4.24|GVGAI)" | grep -v grep
fi
