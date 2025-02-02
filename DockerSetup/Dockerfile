# Use the official osrm backend image
FROM osrm/osrm-backend:latest

# Set the working directory inside the container
WORKDIR /data

# Copy the Netherlands map file from the host to the container
COPY netherlands-latest.osm.pbf /data/netherlands-latest.osm.pbf

# Preprocess the map data (extract and contract)
RUN osrm-extract -p /opt/car.lua /data/netherlands-latest.osm.pbf && \
    osrm-contract /data/netherlands-latest.osrm

# Start the osrm server
CMD ["osrm-routed", "/data/netherlands-latest.osrm"]
