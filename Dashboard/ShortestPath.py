#test file van Wester

import pandas as pd
import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import itertools

# Cache the data loading function
@st.cache
def load_and_process_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, encoding="ISO-8859-1")
    elif file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file, engine='openpyxl')
    else:
        return None

# File uploader
fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))

if fl is not None:
    # Load the file using the cached function
    df = load_and_process_data(fl)
    st.write(f"Uploaded file: {fl.name}")
    st.dataframe(df)  # Display the dataframe

    # Check for required columns
    required_columns = ['lat', 'lon', 'name']
    if all(col in df.columns for col in required_columns):
        df_clean = df[['lat', 'lon', 'name']].dropna()  # Keep necessary columns

        # Unique names for the dropdown menu
        unique_names = df_clean['name'].unique()

        # Dropdown to select a specific name
        selected_name = st.selectbox("Select a name", unique_names)

        # Filter data for the selected name
        filtered_data = df_clean[df_clean['name'] == selected_name]
        coordinates = filtered_data[['lat', 'lon', 'name']].values.tolist()

        # Create a color palette (fixed colors for simplicity)
        predefined_colors = itertools.cycle(['blue', 'green', 'red', 'purple', 'orange'])
        colors = {selected_name: next(predefined_colors)}  # Use only one color for the selected name

        # Create a Folium map centered at the first point of the filtered data
        m = folium.Map(location=[coordinates[0][0], coordinates[0][1]], zoom_start=12, tiles='cartodb positron')

        # Use MarkerCluster for efficient rendering of many points
        marker_cluster = MarkerCluster().add_to(m)

        # Add markers to the cluster for the filtered data
        for lat, lon, name in coordinates:
            folium.Marker(
                location=[lat, lon],
                tooltip=f"{name}: ({lat}, {lon})",
                icon=folium.Icon(color=colors[name], icon_color='white')
            ).add_to(marker_cluster)

        # Display the map in Streamlit
        st.title(f"Mapped Points for '{selected_name}'")
        st_folium(m, width=800, height=600)
    else:
        st.error("The file must contain 'latitude', 'longitude', and 'name' columns.")

#TEST