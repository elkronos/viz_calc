import json
from urllib.request import urlopen
import plotly.express as px
import pandas as pd
import numpy as np

def create_choropleth(config):
    """
    Creates a choropleth map using plotly.express and the provided configuration.
    
    Parameters:
    - config (dict): A dictionary containing the configuration parameters for the choropleth. Keys are:
        - data_frame (pd.DataFrame, optional): The data frame containing the data to be plotted.
        - geojson_url (str): The URL to the geojson file containing the geographic boundaries.
        - locations_col (str): The column name in data_frame representing the location codes.
        - color_col (str): The column name in data_frame representing the color values.
        - color_scale (str, optional): The color scale to use for the choropleth. Default is "Viridis".
        - range_color (tuple, optional): The range of color values. Default is determined from data.
        - scope (str, optional): The scope of the choropleth (e.g., "usa"). Default is "usa".
        - labels (dict, optional): A dictionary of labels for the choropleth.
        - output_html (str, optional): The path to save the choropleth as an HTML file (optional).
        - iframe_html (str, optional): The path to save the iframe HTML file for embedding in a Jupyter Notebook (optional).
    
    Returns:
    - fig: The created figure.
    """
    try:
        with urlopen(config['geojson_url']) as response:
            geojson_data = json.load(response)
    except Exception as e:
        print(f"Failed to load GeoJSON data: {e}")
        return None

    fips_codes = [feature['properties']['GEO_ID'][-5:] for feature in geojson_data['features']]
    
    data_frame = pd.DataFrame({
        config['locations_col']: fips_codes,
        config['color_col']: np.random.rand(len(fips_codes)) * 12
    })
    
    range_color = config.get('range_color')
    if range_color is None:
        range_color = [data_frame[config['color_col']].min(), data_frame[config['color_col']].max()]

    # Build the choropleth
    fig = px.choropleth(data_frame, 
        geojson=geojson_data, 
        locations=config['locations_col'], 
        color=config['color_col'],
        color_continuous_scale=config.get('color_scale', "Viridis"),
        range_color=range_color,
        scope=config.get('scope', "usa"),
        labels=config.get('labels')
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Improve the legend
    fig.update_layout(coloraxis_colorbar=dict(
        thicknessmode="pixels", thickness=10,
        lenmode="pixels", len=150,
        yanchor="top", y=0.8,
        ticks="outside", ticksuffix=" %",
        dtick=5
    ))

    # Save as HTML if needed
    if config.get('output_html'):
        fig.write_html(config['output_html'])

    # Save iframe HTML if needed
    if config.get('iframe_html'):
        iframe_content = f"""
        <iframe src="{config['output_html']}" width="800" height="600" title="Choropleth map with plotly" style="border:none"></iframe>
        """
        with open(config['iframe_html'], 'w') as f:
            f.write(iframe_content)
    
    return fig

# Usage of create_choropleth function
create_choropleth({
    'geojson_url': 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json',
    'locations_col': 'fips',
    'color_col': 'unemp',
    'labels': {'unemp': 'unemployment rate'},
    'output_html': "choropleth-map-plotly-python.html",
    'iframe_html': "iframe_choropleth.html"
}).show()
