#!/bin/bash

# Stop the script on any errors
set -e

# Navigate to the DockerSetup directory
cd "$(dirname "$0")"

# Check if the map file exists
if [ ! -f "netherlands-latest.osm.pbf" ]; then
    echo "Error: netherlands-latest.osm.pbf not found in the DockerSetup directory."
    echo "Please ensure the file is present."
    exit 1
fi

# Build the Docker image
echo "Building the OSRM Docker image for the Netherlands map ......"
docker build -t osrm-netherlands .

# Run the osrm server (container)
echo "Starting the OSRM server on port 5000 ......"
docker run -p 5000:5000 osrm-netherlands
