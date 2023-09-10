import folium
import pandas as pd

def create_markermap(data, lat_col, lon_col, data_col, color_col, marker_radius=5, marker_fill_opacity=0.6, marker_color='blue'):
    # Create a map centered at an average location
    center_lat = data[lat_col].mean()
    center_lon = data[lon_col].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

    # Adding markers to the map
    for idx, row in data.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=marker_radius,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=marker_fill_opacity,
            popup=f"{data_col}: {row[data_col]}, {color_col}: {row[color_col]}"
        ).add_to(m)

    return m

# Generating a synthetic data set
data = {
    'State': ['CA', 'TX', 'FL', 'NY', 'IL'],
    'Latitude': [36.7783, 31.9686, 27.9944024, 40.7128, 40.6331],
    'Longitude': [-119.4179, -99.9018, -81.0754657, -74.0060, -89.3995],
    'DataColumn': [300, 200, 400, 500, 100],
    'ColorColumn': [1, 2, 3, 4, 5]
}

df = pd.DataFrame(data)

# Call the function with necessary arguments
create_markermap(df, 'Latitude', 'Longitude', 'DataColumn', 'ColorColumn', marker_radius=10, marker_fill_opacity=0.8, marker_color='red')
