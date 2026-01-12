#!/bin/bash
# Quick script to start Jupyter for VS Code connection
# Run this in your interactive job

# Get node info
NODE=$(hostname -s)
PORT=8888

# Find available port
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    PORT=$((PORT + 1))
done

echo "=========================================="
echo "Starting Jupyter for VS Code"
echo "=========================================="
echo "Node: $NODE"
echo "Port: $PORT"
echo ""
echo "1. Keep this terminal open!"
echo ""
echo "2. On your LOCAL machine, run this SSH tunnel:"
echo "   ssh -L $PORT:$NODE:$PORT anmari@eu-login-32"
echo ""
echo "3. In VS Code, when Jupyter starts, use:"
echo "   http://localhost:$PORT/?token=<TOKEN_FROM_BELOW>"
echo ""
echo "=========================================="
echo ""

cd /cluster/home/anmari/meta_vlas

# Activate the libero virtual environment
if [ -d "meta_vlas/.venv" ]; then
    source meta_vlas/.venv/bin/activate
fi

# Start Jupyter - it will print the token
jupyter notebook --no-browser --ip=$NODE --port=$PORT --allow-root

