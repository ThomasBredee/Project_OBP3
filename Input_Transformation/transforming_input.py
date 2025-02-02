#########################################################
#                                                       #
# Created on: 24/01/2025                                #
# Created by: Lukas                                     #
#                                                       #
# Updated on: 24/01/2025                                #
# Updated by: Dennis                                    #
#                                                       #
#########################################################

import requests
import time
import pandas as pd

class TransformInput:
    def __init__(self, osrm_url="http://localhost:5000", check_road_proximity=True):
        self.osrm_url = osrm_url
        self.session = requests.Session()
        self.check_road_proximity = check_road_proximity

    def execute_validations(self, df):
        print("Starting validation process...")
        df = self.drop_duplicates(df)
        df = self._drop_floats(df)
        df = self._add_underscore(df)
        if self.check_road_proximity:
            df = self._validate_points_near_roads(df)
        return df

    def _validate_points_near_roads(self, df, max_distance_km=1):
        """Check if points in the DataFrame are within a certain distance from a road and log excluded points."""
        # Add a new column to store proximity result
        df['near_road'] = False

        for index, row in df.iterrows():
            lat, lon = row['lat'], row['lon']
            retries = 3  # Number of retries for the request
            point_near_road = False

            for attempt in range(retries):
                try:
                    url = f"{self.osrm_url}/nearest/v1/driving/{lon},{lat}"
                    response = self.session.get(url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if "waypoints" in data and len(data["waypoints"]) > 0:
                            distance_meters = data["waypoints"][0]["distance"]
                            distance_km = distance_meters / 1000  # Convert to kilometers
                            point_near_road = distance_km <= max_distance_km
                            break  # Exit retry loop if successful
                except requests.exceptions.RequestException as e:
                    print(f"Connection failed on attempt {attempt + 1}/{retries} for {row['name']}. Error: {e}")
                    time.sleep(1)  # Reduce sleep time if needed

            # Update the DataFrame with the result
            df.at[index, 'near_road'] = point_near_road

        # Log and exclude points not near any road
        excluded_points = df[~df['near_road']]
        if not excluded_points.empty:
            print("Excluded points not near any road:")
            print(excluded_points[['name', 'lat', 'lon']])

        # Return DataFrame excluding the points that are not near a road
        return df[df['near_road']].drop(columns=['near_road'])

    def drop_duplicates(self, df):
        """Remove duplicate rows based on latitude, longitude, and name."""
        duplicates = df.duplicated(subset=['lat', 'lon', 'name'], keep='first')
        removed_rows = df[duplicates]
        duplicates_removed = duplicates.sum()
        if duplicates_removed > 0:
            print(f"{duplicates_removed} duplicate(s) found and removed.")
            removed_names = removed_rows['name'].tolist()
            print(f"Removed companies: {removed_names}")
        return df[~duplicates].reset_index(drop=True)

    def _drop_floats(self, df):
        """Convert non-float lat/lon values to float and drop rows with invalid lat/lon."""
        converted_count = 0
        removed_count = 0
        for column in ['lat', 'lon']:
            original_values = df[column].copy()
            df.loc[:, column] = pd.to_numeric(df[column], errors='coerce')
            converted_count += ((original_values != df[column]) & df[column].notna()).sum()
        before_removal = len(df)
        df = df.dropna(subset=['lat', 'lon']).reset_index(drop=True)
        removed_count += before_removal - len(df)
        if converted_count > 0:
            print(f"{converted_count} non-float value(s) successfully converted to floats.")
        if removed_count > 0:
            print(f"{removed_count} row(s) removed due to invalid lat/lon values.")
        else:
            print("All lat and lon values are already valid floats.")
        return df

    def _add_underscore(self, df):
        """Add an underscore and a count suffix to duplicate names."""
        df['name'] = df.groupby('name').cumcount().add(1).astype(str).radd(df['name'] + "_")
        return df

    def _slice_df(self, df1, df2):
        unique_names = df1['name'].unique()

        # Filter rows in df2 where the index matches unique names
        filtered_rows = df2.loc[df2.index.isin(unique_names)]

        # Filter columns in df2 where the column names match unique names
        filtered_df2 = filtered_rows.loc[:, filtered_rows.columns.isin(unique_names)]
        return filtered_df2
