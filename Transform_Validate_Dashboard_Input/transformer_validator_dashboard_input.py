import requests
import time

class ValidateInput:
    def __init__(self, osrm_url="http://localhost:5000"):
        self.osrm_url = osrm_url
        self.session = requests.Session()

    def is_point_near_road(self, lat, lon, max_distance_km=2, retries=3):
        """Check if a point is within a certain distance from a road."""
        for attempt in range(retries):
            try:
                url = f"{self.osrm_url}/nearest/v1/driving/{lon},{lat}"
                response = self.session.get(url, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    if "waypoints" in data and len(data["waypoints"]) > 0:
                        # Distance from the nearest road in meters
                        distance_meters = data["waypoints"][0]["distance"]
                        distance_km = distance_meters / 1000  # Convert to kilometers

                        if distance_km <= max_distance_km:
                            return True  # Within the specified range of a road
                return False  # Point too far from a road or invalid

            except requests.ConnectionError:
                print(f"Connection failed. Retrying {attempt + 1}/{retries}...")
                time.sleep(5)

        return False  # If all retries fail, assume it's not feasible



def main():
    # Initialize the class
    validator = ValidateInput(osrm_url="http://localhost:5000")

    # Example coordinates
    coordinates = [
        (52.38247865775596, 4.868180384630428),  # Example: Amsterdam (should return True)
        (52.63547187447808, 5.05045887081477)  # Example: Likely in the sea (should return False)
    ]

    for lat, lon in coordinates:
        is_land = validator.is_point_near_road(lat, lon)
        print(f"Coordinate ({lat}, {lon}) is on land: {is_land}")


if __name__ == "__main__":
    main()
