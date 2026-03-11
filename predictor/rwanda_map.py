import folium
import pandas as pd
from folium import plugins
import json
import os
import random

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

# Function to calculate center point of a polygon from GeoJSON coordinates
def calculate_district_center(geojson_coords):
    """Calculate the center point of a polygon from GeoJSON coordinates"""
    if not geojson_coords or len(geojson_coords) == 0:
        return [0, 0]
    
    # GeoJSON coordinates are in format [lon, lat]
    lats = [coord[1] for coord in geojson_coords]
    lons = [coord[0] for coord in geojson_coords]
    
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    return [center_lat, center_lon]


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
    
    # Generate distinct colors for each district
    def generate_district_colors(num_districts):
        """Generate visually distinct colors for districts"""
        colors = []
        # Use a predefined set of distinct colors
        base_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
            '#F8B739', '#52B788', '#E76F51', '#8E44AD', '#3498DB',
            '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C',
            '#34495E', '#E67E22', '#95A5A6', '#16A085', '#27AE60',
            '#2980B9', '#8E44AD', '#2C3E50', '#F1C40F', '#D35400'
        ]
        
        # If we need more colors than available, generate random ones
        for i in range(num_districts):
            if i < len(base_colors):
                colors.append(base_colors[i])
            else:
                # Generate random distinct colors
                hue = random.randint(0, 360)
                saturation = random.randint(60, 90)
                lightness = random.randint(40, 70)
                colors.append(f'hsl({hue}, {saturation}%, {lightness}%)')
        
        return colors
    
    # Add district boundaries from GeoJSON
    geojson_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dummy-data', 'rwanda_districts.geojson')
    if os.path.exists(geojson_path):
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Generate colors for each district
        district_colors = generate_district_colors(len(geojson_data['features']))
        
        # Create a mapping of district names to colors and coordinates
        color_mapping = {}
        coordinate_mapping = {}
        
        for i, feature in enumerate(geojson_data['features']):
            district_name = feature['properties'].get('name', '')
            if district_name:
                color_mapping[district_name] = district_colors[i]
                
                # Calculate center coordinates from GeoJSON
                coords = feature['geometry'].get('coordinates', [[[]]])[0]
                if coords:
                    coordinate_mapping[district_name] = calculate_district_center(coords)
        
        # Style function for districts with different colors and bold borders
        def style_function(feature):
            district_name = feature['properties'].get('name', '')
            fill_color = color_mapping.get(district_name, '#E0E0E0')
            
            # Get client count for this district
            client_count = 0
            if district_name:
                matching_districts = district_counts[district_counts['district'] == district_name]
                if not matching_districts.empty:
                    client_count = matching_districts.iloc[0]['client_count']
            
            # Adjust opacity based on client count
            fill_opacity = min(0.8, 0.2 + (client_count / district_counts['client_count'].max()) * 0.6)
            
            return {
                'fillColor': fill_color,
                'color': '#2C3E50',  # Dark bold borders
                'weight': 3,  # Bold borders
                'fillOpacity': fill_opacity,
                'dashArray': None,  # Solid lines instead of dashed
                'opacity': 1.0
            }
        
        # Highlight function for hover effect
        def highlight_function(feature):
            return {
                'fillColor': '#FFD700',  # Gold highlight
                'color': '#000000',  # Black border on hover
                'weight': 4,  # Even bolder on hover
                'fillOpacity': 0.9,
                'dashArray': None
            }
        
        # Add GeoJSON with enhanced styling
        folium.GeoJson(
            geojson_data,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['name'],
                aliases=['District:'],
                localize=True,
                sticky=False,
                labels=True,
                style="""
                    background-color: white;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px 3px 3px rgba(0,0,0,0.4);
                    font-size: 12px;
                    font-weight: bold;
                """
            )
        ).add_to(m)
        
        # Add district boundary labels with better styling
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
                    
                    # Get client count for this district
                    client_count = 0
                    if district_name:
                        matching_districts = district_counts[district_counts['district'] == district_name]
                        if not matching_districts.empty:
                            client_count = matching_districts.iloc[0]['client_count']
                    
                    # Add district name label with enhanced styling
                    if district_name:
                        folium.Marker(
                            location=[center_lat, center_lon],
                            icon=folium.DivIcon(html=f'''
                                <div style="font-size: 11px; color: #2C3E50; font-weight: bold; 
                                text-align: center; background-color: rgba(255,255,255,0.95); 
                                padding: 4px 6px; border-radius: 4px; border: 2px solid #2C3E50;
                                box-shadow: 2px 2px 4px rgba(0,0,0,0.3); white-space: nowrap;">
                                    {district_name}<br>
                                    <span style="font-size: 9px; color: #E74C3C;">{client_count} clients</span>
                                </div>''')
                        ).add_to(m)
    
    # Find min and max counts for circle marker scaling
    min_count = district_counts['client_count'].min()
    max_count = district_counts['client_count'].max()
    
    # Add circle markers for each district to show user count
    for _, row in district_counts.iterrows():
        district = row['district']
        count = row['client_count']
        
        # Use coordinates from RWANDA_DISTRICTS if available
        if district in RWANDA_DISTRICTS:
            coords = RWANDA_DISTRICTS[district]
            
            # Determine circle color based on client count (gradient from green to red)
            if max_count - min_count > 0:
                normalized = (count - min_count) / (max_count - min_count)
            else:
                normalized = 0.5
            
            # Create color gradient for circles
            if normalized < 0.33:
                circle_color = '#27AE60'  # Green
            elif normalized < 0.66:
                circle_color = '#F39C12'  # Orange  
            else:
                circle_color = '#E74C3C'  # Red
            
            # Calculate circle radius based on client count
            radius = max(8, min(25, (count / max_count) * 20 + 8))
            
            # Add circle marker with enhanced styling
            folium.CircleMarker(
                location=coords,
                radius=radius,
                popup=f"""
                <div style="font-family: Arial, sans-serif; padding: 10px;">
                    <h4 style="margin: 0 0 8px 0; color: #2C3E50;">{district}</h4>
                    <p style="margin: 4px 0;"><strong>Clients:</strong> {count}</p>
                    <p style="margin: 4px 0;"><strong>Density:</strong> {normalized:.1%}</p>
                </div>
                """,
                tooltip=f"{district}: {count} clients",
                color='#2C3E50',  # Dark border
                fill=True,
                fillColor=circle_color,
                fillOpacity=0.7,
                weight=2,
                opacity=0.9
            ).add_to(m)
            
            # Add client count number in the center of circle
            folium.Marker(
                location=coords,
                icon=folium.DivIcon(html=f'''
                    <div style="font-size: 12px; color: white; font-weight: bold; 
                    text-align: center; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
                    pointer-events: none;">
                        {count}
                    </div>''')
            ).add_to(m)
    
    # Add legend with enhanced styling including circle information
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 300px; height: auto; 
                background-color: white; border:3px solid #2C3E50; z-index:9999; 
                font-size:14px; padding: 15px; border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
        <h6 style="margin-top: 0; color: #2C3E50; font-weight: bold;">
            <i class="fas fa-map"></i> Rwanda District Map
        </h6>
        <hr style="margin: 10px 0; border-color: #2C3E50;">
        <p style="margin: 8px 0; font-size: 12px;"><i class="fas fa-palette"></i> Each district has a unique color</p>
        <p style="margin: 8px 0; font-size: 12px;"><i class="fas fa-border-style"></i> Bold borders show district boundaries</p>
        <p style="margin: 8px 0; font-size: 12px;"><i class="fas fa-circle"></i> Circles show client density</p>
        <p style="margin: 8px 0; font-size: 12px;"><i class="fas fa-mouse-pointer"></i> Hover for details</p>
        <hr style="margin: 10px 0; border-color: #2C3E50;">
        <p style="margin: 5px 0; font-size: 11px;"><i class="fas fa-circle" style="color: #27AE60"></i> Low Density</p>
        <p style="margin: 5px 0; font-size: 11px;"><i class="fas fa-circle" style="color: #F39C12"></i> Medium Density</p>
        <p style="margin: 5px 0; font-size: 11px;"><i class="fas fa-circle" style="color: #E74C3C"></i> High Density</p>
        <hr style="margin: 10px 0; border-color: #2C3E50;">
        <p style="margin: 5px 0; font-size: 12px;"><b>Total Districts:</b> {}</p>
        <p style="margin: 5px 0; font-size: 12px;"><b>Total Clients:</b> {}</p>
        <p style="margin: 5px 0; font-size: 12px;"><b>Fill Opacity:</b> Based on client count</p>
    </div>
    '''.format(len(district_counts), district_counts['client_count'].sum())
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Return map as HTML
    return m._repr_html_()
