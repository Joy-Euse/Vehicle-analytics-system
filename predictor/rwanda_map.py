import folium
import pandas as pd
from folium import plugins
import json
import os

# Rwanda districts with approximate coordinates (center of each district)
RWANDA_DISTRICTS = {
    "Rusizi": [-2.504, 29.015],
    "Rubavu": [-1.491, 29.246],
    "Burera": [-1.576, 29.632],
    "Gicumbi": [-1.633, 29.896],
    "Musanze": [-1.494, 29.632],
    "Nyabihu": [-1.969, 29.446],
    "Gakenke": [-1.846, 29.999],
    "Karongi": [-2.062, 29.258],
    "Nyarugenge": [-1.949, 30.060],
    "Gasabo": [-1.952, 30.194],
    "Kicukiro": [-1.981, 30.032],
    "Rwamagana": [-2.129, 30.460],
    "Ngoma": [-2.210, 30.243],
    "Kayonza": [-2.128, 30.722],
    "Bugesera": [-2.352, 30.413],
    "Nyaruguru": [-2.639, 29.280],
    "Nyamagabe": [-2.601, 29.560],
    "Huye": [-2.597, 29.745],
    "Gisagara": [-2.795, 29.631],
    "Nyanza": [-2.663, 29.930],
    "Muhanga": [-1.947, 30.134],
}


def create_rwanda_map_with_districts(df):
    """
    Create an interactive map of Rwanda showing client distribution by district.
    
    Args:
        df: DataFrame with vehicle client data
    
    Returns:
        HTML string of the map
    """
    # Count clients by district
    district_counts = df['district'].value_counts().reset_index()
    district_counts.columns = ['district', 'client_count']
    
    # Create base map centered on Rwanda
    rwanda_center = [-1.95, 29.87]
    m = folium.Map(
        location=rwanda_center,
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Add district boundaries from GeoJSON
    geojson_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dummy-data', 'rwanda_districts.geojson')
    if os.path.exists(geojson_path):
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        folium.GeoJson(
            geojson_data,
            style_function=lambda x: {
                'fillColor': '#f0f0f0',
                'color': '#1a3a5c',
                'weight': 4,
                'fillOpacity': 0.05,
                'dashArray': '5, 5'
            },
            highlight_function=lambda x: {
                'fillColor': '#ffff00',
                'color': '#000000',
                'weight': 5,
                'fillOpacity': 0.15
            }
        ).add_to(m)
        
        # Add district boundary labels
        for feature in geojson_data['features']:
            if 'properties' in feature and 'geometry' in feature:
                district_name = feature['properties'].get('name', '')
                coords = feature['geometry'].get('coordinates', [[[]]])[0]
                if coords:
                    # Calculate center of boundary
                    lats = [c[1] for c in coords]
                    lons = [c[0] for c in coords]
                    center_lat = sum(lats) / len(lats) if lats else 0
                    center_lon = sum(lons) / len(lons) if lons else 0
                    
                    # Add district name label on boundary
                    if district_name:
                        folium.Marker(
                            location=[center_lat, center_lon],
                            icon=folium.DivIcon(html=f'''
                                <div style="font-size: 10px; color: #1a3a5c; font-weight: bold; 
                                text-align: center; background-color: rgba(255,255,255,0.9); 
                                padding: 3px 5px; border-radius: 3px; border: 1px solid #1a3a5c;">
                                    {district_name}
                                </div>''')
                        ).add_to(m)
    
    # Find min and max counts for color scaling
    min_count = district_counts['client_count'].min()
    max_count = district_counts['client_count'].max()
    
    # Add markers for each district
    for _, row in district_counts.iterrows():
        district = row['district']
        count = row['client_count']
        
        if district in RWANDA_DISTRICTS:
            coords = RWANDA_DISTRICTS[district]
            
            # Color based on client count (gradient from green to red)
            if max_count - min_count > 0:
                normalized = (count - min_count) / (max_count - min_count)
            else:
                normalized = 0.5
            
            # Create color gradient
            if normalized < 0.33:
                color = 'green'
            elif normalized < 0.66:
                color = 'orange'
            else:
                color = 'red'
            
            # Add circle marker
            folium.CircleMarker(
                location=coords,
                radius=max(5, min(20, count / 2)),
                popup=f"<b>{district}</b><br>Clients: {count}",
                tooltip=f"{district}: {count} clients",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
            
            # Add text label
            folium.Marker(
                location=coords,
                icon=folium.DivIcon(html=f'''
                    <div style="font-size: 12px; color: black; font-weight: bold; 
                    text-align: center; background-color: rgba(255,255,255,0.8); 
                    padding: 2px; border-radius: 3px;">
                        {count}
                    </div>
                ''')
            ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 250px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;">
        <p style="margin-top: 0;"><b>Rwanda Vehicle Client Distribution</b></p>
        <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:green"></i> Low Distribution</p>
        <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:orange"></i> Medium Distribution</p>
        <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:red"></i> High Distribution</p>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0;"><b>Total Districts:</b> {}</p>
        <p style="margin: 5px 0;"><b>Total Clients:</b> {}</p>
    </div>
    '''.format(len(district_counts), district_counts['client_count'].sum())
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Return map as HTML
    return m._repr_html_()
