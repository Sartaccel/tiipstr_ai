#!/usr/bin/env bash

echo "Starting watcher..."
python watcher.py &

echo "Starting API..."
python -m uvicorn api:app --host 0.0.0.0 --port $PORT
