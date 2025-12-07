#!/bin/bash

echo "Starting Face Search Application..."

# Create necessary directories
mkdir -p uploads uploads-link chroma_db

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker environment"
    # In Docker, bind to all interfaces
    export FLASK_HOST=0.0.0.0
else
    echo "Running locally"
    # Locally, can bind to localhost
    export FLASK_HOST=127.0.0.1
fi

export FLASK_PORT=5000
export FLASK_ENV=production

echo "Starting Flask app on $FLASK_HOST:$FLASK_PORT"

# Start the Flask application
python app.py