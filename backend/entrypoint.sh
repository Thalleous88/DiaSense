#!/bin/bash
set -e

if [ ! -f /app/ensemble_model.pkl ]; then
    echo "No model artifacts found. Running train.py..."
    python train.py
    echo "Training complete. Starting API server..."
else
    echo "Model artifacts found. Skipping training."
fi

exec "$@"
