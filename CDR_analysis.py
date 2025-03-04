#Import the neccessary modules
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster, AntPath
from folium import plugins
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from datetime import datetime
import math
from shapely.geometry import LineString
import geopandas as gpd
from collections import Counter
import warnings
import calendar
warnings.filterwarnings("ignore")

#LOAD THE DATA
def load_real_cdr_data(file_path):
    """
    Load the actual CDR data with the specified column structure
    """
    try:
        # Load the data with the correct column names
        df = pd.read_csv(file_path)
        
        # Standardize column names for compatibility with the analysis functions
        column_mapping = {
            'User_ID': 'user_id',
            'Date_Time': 'timestamp',
            'Latitude': 'lat',
            'Longitude': 'lon',
            'Tower_ID': 'tower_id',
            'Type': 'call_type',
            'Duration': 'duration',
            'Cost': 'cost',
            'Data_Usage_MB': 'data_usage',
            'Device_Type': 'device_type',
            'Activity_Flag': 'activity_flag',
            'Region': 'region'
        }
        
        df = df.rename(columns=column_mapping)
        df['activity_flag'] = df['activity_flag'].fillna("Standard")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract date components for time analysis
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['week'] = df['timestamp'].dt.isocalendar().week
        
        # Add day names for better readability
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                     4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        df['day_name'] = df['day_of_week'].map(day_names)
        
        # Add month names
        month_names = {i: calendar.month_name[i] for i in range(1, 13)}
        df['month_name'] = df['month'].map(month_names)
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

 #IDENTIFY TOURIST SITES       
def identify_tourists(df):
    """
    Identify likely tourists in the CDR data based on:
    1. Roaming activity
    2. International call patterns
    3. Short-term presence in tourist areas
    """
    # Initialize an empty set to hold tourist user IDs
    tourist_ids = set()
    
    # 1. Users with roaming activity are likely tourists
    if 'activity_flag' in df.columns:
        roaming_users = df[df['activity_flag'] == 'roaming']['user_id'].unique()
        tourist_ids.update(roaming_users)
        print(f"Identified {len(roaming_users)} users with roaming activity")
    
    # 2. Users with international calls
    if 'activity_flag' in df.columns:
        intl_call_users = df[df['activity_flag'] == 'international call']['user_id'].unique()
        tourist_ids.update(intl_call_users)
        print(f"Identified {len(intl_call_users)} users with international calls")
    
    # 3. Short-term visitors (active for less than 14 days)
    user_duration = df.groupby('user_id')['date'].agg(['min', 'max'])
    user_duration['days_active'] = (user_duration['max'] - user_duration['min']).dt.days + 1
    short_term_users = user_duration[user_duration['days_active'] <= 14].index.tolist()
    
    # Only count as tourists if they've been seen in tourist areas
    # List of known tourist regions/areas in Rwanda (modify as needed)
    tourist_regions = [
        'Volcanoes National Park', 'Nyungwe Forest', 'Akagera National Park',
        'Lake Kivu', 'Kigali', 'Musanze', 'Gisenyi', 'Rubavu'
    ]
    
    # Find regions in our dataset that contain any of these tourist areas
    tourist_region_matches = []
    if 'region' in df.columns:
        for region in df['region'].unique():
            if any(tourist_area.lower() in str(region).lower() for tourist_area in tourist_regions):
                tourist_region_matches.append(region)
    
    # Filter short-term users who visited tourist areas
    if tourist_region_matches and 'region' in df.columns:
        tourist_area_visitors = df[df['region'].isin(tourist_region_matches)]['user_id'].unique()
        tourist_short_term = set(short_term_users) & set(tourist_area_visitors)
        tourist_ids.update(tourist_short_term)
        print(f"Identified {len(tourist_short_term)} short-term visitors to tourist areas")
    else:
        # If no region info, just use short-term status
        tourist_ids.update(short_term_users)
        print(f"Identified {len(short_term_users)} short-term visitors (no region filter)")
    
    print(f"Total identified tourists: {len(tourist_ids)}")
    return list(tourist_ids)

#CREATE HEATMAPS
def create_tourist_heatmap(df, tourist_ids, zoom_start=8):
    """
    Create a heatmap showing the density of tourist activity
    """
    # Filter for tourist activities
    tourist_df = df[df['user_id'].isin(tourist_ids)]
    
    # Create a map centered on Rwanda
    m = folium.Map(location=[-1.9403, 29.8739], zoom_start=zoom_start)
    
    # Add heatmap layer
    heat_data = [[row['lat'], row['lon']] for _, row in tourist_df.iterrows()]
    HeatMap(heat_data, radius=15, max_zoom=13).add_to(m)
    
    # Add title
    title_html = '''
             <h3 align="center" style="font-size:16px"><b>Tourist Activity Heatmap in Rwanda</b></h3>
             <p align="center">Based on CDR data analysis</p>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m
def analyze_tourist_hotspots(df, tourist_ids, eps=0.01, min_samples=5):
    """
    Identify tourist hotspots using DBSCAN clustering algorithm
    """
    # Filter for tourist activities
    tourist_df = df[df['user_id'].isin(tourist_ids)]
    
    # Extract coordinates
    coords = tourist_df[['lat', 'lon']].values
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    tourist_df['cluster'] = clustering.labels_
    
    # Count points in each cluster
    cluster_counts = Counter(tourist_df[tourist_df['cluster'] >= 0]['cluster'])
    
    # Calculate cluster centers and additional metrics
    hotspots = []
    for cluster_id, count in cluster_counts.items():
        cluster_data = tourist_df[tourist_df['cluster'] == cluster_id]
        
        # Calculate center
        center_lat = cluster_data['lat'].mean()
        center_lon = cluster_data['lon'].mean()
        
        # Get the most common region if available
        if 'region' in cluster_data.columns:
            most_common_region = cluster_data['region'].mode()[0] if not cluster_data['region'].empty else 'Unknown'
        else:
            most_common_region = 'Unknown'
        
        # Calculate unique visitors
        unique_visitors = cluster_data['user_id'].nunique()
        
        # Calculate average time spent (if multiple records per user)
        avg_time_spent = 0
        if 'timestamp' in cluster_data.columns:
            # Group by user to find first and last timestamp
            user_times = cluster_data.groupby('user_id')['timestamp'].agg(['min', 'max'])
            time_diffs = (user_times['max'] - user_times['min']).dt.total_seconds() / 3600  # in hours
            avg_time_spent = time_diffs.mean() if not pd.isna(time_diffs.mean()) else 0
        
        # Calculate most common activity types
        if 'call_type' in cluster_data.columns:
            activity_types = cluster_data['call_type'].value_counts().to_dict()
        else:
            activity_types = {}
        
        # Calculate temporal patterns
        if 'hour' in cluster_data.columns:
            busy_hours = cluster_data['hour'].value_counts().nlargest(3).index.tolist()
        else:
            busy_hours = []
            
        if 'day_name' in cluster_data.columns:
            busy_days = cluster_data['day_name'].value_counts().nlargest(3).index.tolist()
        else:
            busy_days = []
        
        hotspots.append({
            'cluster_id': int(cluster_id),
            'count': count,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'region': most_common_region,
            'unique_visitors': unique_visitors,
            'avg_time_spent': avg_time_spent,
            'activity_types': activity_types,
            'busy_hours': busy_hours,
            'busy_days': busy_days
        })
    
    # Convert to DataFrame and sort by count
    hotspots_df = pd.DataFrame(hotspots).sort_values('count', ascending=False)
    
    return hotspots_df, tourist_df

def visualize_tourist_hotspots(hotspots_df, tourist_df_with_clusters, zoom_start=8):
    """
    Create a map visualization of the identified tourist hotspots
    """
    # Create a map centered on Rwanda
    m = folium.Map(location=[-1.9403, 29.8739], zoom_start=zoom_start)
    
    # Add a marker for each hotspot
    for idx, hotspot in hotspots_df.iterrows():
        # Scale circle size based on count
        radius = math.sqrt(hotspot['count']) * 50
        
        # Create popup content
        popup_content = f"""
        <h4>Tourist Hotspot #{hotspot['cluster_id']+1}</h4>
        <b>Region:</b> {hotspot['region']}<br>
        <b>Visitors:</b> {hotspot['unique_visitors']}<br>
        <b>Avg Time Spent:</b> {hotspot['avg_time_spent']:.2f} hours<br>
        <b>Popular Times:</b> {', '.join([str(h) + ':00' for h in hotspot['busy_hours']][:3])}<br>
        <b>Popular Days:</b> {', '.join(hotspot['busy_days'][:3])}<br>
        <b>Total Activities:</b> {hotspot['count']}
        """
        
        # Add circle for hotspot
        folium.Circle(
            location=[hotspot['center_lat'], hotspot['center_lon']],
            radius=radius,
            popup=folium.Popup(popup_content, max_width=300),
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.4,
            tooltip=f"Tourist Hotspot: {hotspot['region']} ({hotspot['unique_visitors']} visitors)"
        ).add_to(m)
        
        # Add label
        folium.Marker(
            location=[hotspot['center_lat'], hotspot['center_lon']],
            icon=folium.DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html=f'<div style="font-size: 14pt; color: white; font-weight: bold; background-color: rgba(255,0,0,0.7); border-radius: 10px; width: 20px; height: 20px; text-align: center; line-height: 20px;">{idx+1}</div>'
            )
        ).add_to(m)
    
    # Add title
    title_html = '''
             <h3 align="center" style="font-size:16px"><b>Tourist Hotspots in Rwanda</b></h3>
             <p align="center">Circle size represents activity volume</p>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def analyze_tourist_movement_patterns(df, tourist_ids):
    """
    Analyze movement patterns of tourists between major locations
    """
    # Filter for tourist activities
    tourist_df = df[df['user_id'].isin(tourist_ids)].copy()
    
    # Sort by user and timestamp
    tourist_df = tourist_df.sort_values(['user_id', 'timestamp'])
    
    # Create sequence of locations for each user
    # Round coordinates to create location zones
    tourist_df['lat_round'] = tourist_df['lat'].round(2)
    tourist_df['lon_round'] = tourist_df['lon'].round(2)
    tourist_df['location_id'] = tourist_df['lat_round'].astype(str) + "_" + tourist_df['lon_round'].astype(str)
    
    # Get transitions between locations
    tourist_df['next_location'] = tourist_df.groupby('user_id')['location_id'].shift(-1)
    tourist_df['same_user_next'] = tourist_df.groupby('user_id')['user_id'].shift(-1) == tourist_df['user_id']
    
    # Filter for valid transitions (same user, different location)
    transitions = tourist_df[
        (tourist_df['same_user_next']) & 
        (tourist_df['location_id'] != tourist_df['next_location']) & 
        (~tourist_df['next_location'].isna())
    ].copy()
    
    # Add location coordinates
    transitions['from_lat'] = transitions['lat']
    transitions['from_lon'] = transitions['lon']
    
    # Get coordinates for next location
    loc_coords = tourist_df[['location_id', 'lat', 'lon']].drop_duplicates('location_id')
    loc_dict = dict(zip(loc_coords['location_id'], zip(loc_coords['lat'], loc_coords['lon'])))
    
    # Add destination coordinates
    transitions['to_lat'] = transitions['next_location'].map(lambda x: loc_dict.get(x, (np.nan, np.nan))[0])
    transitions['to_lon'] = transitions['next_location'].map(lambda x: loc_dict.get(x, (np.nan, np.nan))[1])
    
    # Drop rows with missing coordinates
    transitions = transitions.dropna(subset=['from_lat', 'from_lon', 'to_lat', 'to_lon'])
    
    # Count location-to-location transitions
    location_flows = transitions.groupby(['location_id', 'next_location']).size().reset_index(name='flow_count')
    location_flows = location_flows.sort_values('flow_count', ascending=False)
    
    # Add coordinates to location flows
    location_flows['from_lat'] = location_flows['location_id'].map(lambda x: loc_dict.get(x, (np.nan, np.nan))[0])
    location_flows['from_lon'] = location_flows['location_id'].map(lambda x: loc_dict.get(x, (np.nan, np.nan))[1])
    location_flows['to_lat'] = location_flows['next_location'].map(lambda x: loc_dict.get(x, (np.nan, np.nan))[0])
    location_flows['to_lon'] = location_flows['next_location'].map(lambda x: loc_dict.get(x, (np.nan, np.nan))[1])
    
    # Create region flows only if region column exists
    if 'region' in tourist_df.columns:
        # Get region for each location
        loc_region = tourist_df[['location_id', 'region']].drop_duplicates('location_id')
        region_dict = dict(zip(loc_region['location_id'], loc_region['region']))
        
        # Add region information to transitions
        transitions['from_region'] = transitions['location_id'].map(region_dict)
        transitions['to_region'] = transitions['next_location'].map(region_dict)
        
        # Add region information to location_flows
        location_flows['from_region'] = location_flows['location_id'].map(region_dict)
        location_flows['to_region'] = location_flows['next_location'].map(region_dict)
        
        # Count region-to-region transitions
        region_flows = transitions.groupby(['from_region', 'to_region']).size().reset_index(name='flow_count')
        region_flows = region_flows[region_flows['from_region'] != region_flows['to_region']]
        region_flows = region_flows.sort_values('flow_count', ascending=False)
    else:
        region_flows = None
        # Add empty region columns to location_flows to prevent KeyError
        location_flows['from_region'] = None
        location_flows['to_region'] = None
    
    return location_flows, region_flows, transitions

def visualize_tourist_flows(location_flows, region_flows=None, zoom_start=8):
    """
    Visualize the movement of tourists between locations with directional arrows
    """
    # Create a map centered on Rwanda
    m = folium.Map(location=[-1.9403, 29.8739], zoom_start=zoom_start)
    
    # Add layer for region flows if available
    if region_flows is not None and len(region_flows) > 0 and 'from_region' in location_flows.columns:
        # Calculate region centroids
        region_centroids = {}
        
        for _, flow in region_flows.iterrows():
            from_region = flow['from_region']
            to_region = flow['to_region']
            
            # Skip if missing regions
            if pd.isna(from_region) or pd.isna(to_region):
                continue
            
            # Calculate centroids if not already done
            if from_region not in region_centroids:
                from_rows = location_flows[location_flows['from_region'] == from_region]
                if len(from_rows) > 0:
                    region_centroids[from_region] = (
                        from_rows['from_lat'].mean(),
                        from_rows['from_lon'].mean()
                    )
            
            if to_region not in region_centroids:
                to_rows = location_flows[location_flows['to_region'] == to_region]
                if len(to_rows) > 0:
                    region_centroids[to_region] = (
                        to_rows['to_lat'].mean(),
                        to_rows['to_lon'].mean()
                    )
        
        # Create a feature group for region flows
        region_group = folium.FeatureGroup(name="Region-to-Region Flows")
        m.add_child(region_group)
        
        # Normalize flows for width
        max_region_flow = region_flows['flow_count'].max()
        min_region_flow = region_flows['flow_count'].min()
        
        # Add lines for top region flows
        for _, flow in region_flows.head(20).iterrows():
            from_region = flow['from_region']
            to_region = flow['to_region']
            
            # Skip if missing data
            if (from_region not in region_centroids or 
                to_region not in region_centroids or 
                pd.isna(from_region) or pd.isna(to_region)):
                continue
            
            # Get coordinates
            from_lat, from_lon = region_centroids[from_region]
            to_lat, to_lon = region_centroids[to_region]
            
            # Calculate line width based on flow
            if max_region_flow == min_region_flow:
                width = 3
            else:
                width = 1 + 7 * (flow['flow_count'] - min_region_flow) / (max_region_flow - min_region_flow)
            
            # Create line
            line = folium.PolyLine(
                locations=[[from_lat, from_lon], [to_lat, to_lon]],
                weight=width,
                color='blue',
                opacity=0.6,
                popup=f"From: {from_region}<br>To: {to_region}<br>Flow: {flow['flow_count']}"
            )
            region_group.add_child(line)
            
            # Add origin marker
            folium.CircleMarker(
                location=[from_lat, from_lon],
                radius=5,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.7,
                popup=f"Origin: {from_region}"
            ).add_to(region_group)
            
            # Add destination marker
            folium.CircleMarker(
                location=[to_lat, to_lon],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup=f"Destination: {to_region}"
            ).add_to(region_group)
            
            # Add directional arrow indicators
            # Calculate arrow positions (multiple arrows along the path)
            num_arrows = min(3, max(1, int(width)))
            for i in range(1, num_arrows + 1):
                # Position arrows along the line
                arrow_pos = i / (num_arrows + 1)
                arrow_lat = from_lat + arrow_pos * (to_lat - from_lat)
                arrow_lon = from_lon + arrow_pos * (to_lon - from_lon)
                
                # Calculate arrow direction angle
                dx = to_lon - from_lon
                dy = to_lat - from_lat
                angle = math.degrees(math.atan2(dy, dx))
                
                # Add arrow using FontAwesome icon with rotation
                folium.Marker(
                    location=[arrow_lat, arrow_lon],
                    icon=folium.DivIcon(
                        icon_size=(20, 20),
                        icon_anchor=(10, 10),
                        html=f'<div style="font-size: {10 + width}px; color: blue; transform: rotate({angle}deg);">➔</div>'
                    )
                ).add_to(region_group)
    
    # Always add location-to-location flows
    location_group = folium.FeatureGroup(name="Location-to-Location Flows")
    m.add_child(location_group)
    
    # Normalize flows for width
    max_flow = location_flows['flow_count'].max()
    min_flow = location_flows['flow_count'].min()
    
    # Add lines for top location flows
    for _, flow in location_flows.head(50).iterrows():
        # Skip if missing data
        if pd.isna(flow['from_lat']) or pd.isna(flow['to_lat']):
            continue
        
        # Calculate line width based on flow
        if max_flow == min_flow:
            width = 2
        else:
            width = 1 + 5 * (flow['flow_count'] - min_flow) / (max_flow - min_flow)
        
        # Get coordinates
        from_lat = flow['from_lat']
        from_lon = flow['from_lon']
        to_lat = flow['to_lat']
        to_lon = flow['to_lon']
        
        # Create line
        line = folium.PolyLine(
            locations=[[from_lat, from_lon], [to_lat, to_lon]],
            weight=width,
            color='red',
            opacity=0.6,
            popup=f"Flow Count: {flow['flow_count']}"
        )
        location_group.add_child(line)
        
        # Add arrowheads to indicate direction
        # Calculate midpoint
        mid_lat = (from_lat + to_lat) / 2
        mid_lon = (from_lon + to_lon) / 2
        
        # Calculate arrow direction angle
        dx = to_lon - from_lon
        dy = to_lat - from_lat
        angle = math.degrees(math.atan2(dy, dx))
        
        # Add arrow marker
        folium.Marker(
            location=[mid_lat, mid_lon],
            icon=folium.DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html=f'<div style="font-size: {10 + width}px; color: red; transform: rotate({angle}deg);">➔</div>'
            )
        ).add_to(location_group)
        
        # Add small markers for origin and destination
        folium.CircleMarker(
            location=[from_lat, from_lon],
            radius=3,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.7,
            popup="Origin"
        ).add_to(location_group)
        
        folium.CircleMarker(
            location=[to_lat, to_lon],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup="Destination"
        ).add_to(location_group)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend for direction indicators
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 120px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white; padding: 10px;
                border-radius: 5px;">
        <p><b>Direction Legend</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> Origin</p>
        <p><i class="fa fa-circle" style="color:red"></i> Destination</p>
        <p><span style="color:blue;">➔</span> Flow Direction</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
             <h3 align="center" style="font-size:16px"><b>Tourist Movement Patterns in Rwanda</b></h3>
             <p align="center">Line thickness represents volume of movement, arrows show direction</p>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m


from folium.plugins import HeatMap, MarkerCluster, HeatMapWithTime, TimestampedGeoJson
import seaborn as sns

def analyze_temporal_patterns(df, tourist_ids):
    """
    Analyze temporal patterns of tourist activity
    """
    # Filter for tourist activities
    tourist_df = df[df['user_id'].isin(tourist_ids)]
    
    # Prepare results dictionary
    results = {}
    
    # 1. Activity by hour of day
    if 'hour' in tourist_df.columns:
        hourly_activity = tourist_df.groupby('hour').size()
        results['hourly_activity'] = hourly_activity
    
    # 2. Activity by day of week
    if 'day_name' in tourist_df.columns and 'day_of_week' in tourist_df.columns:
        # Sort by day of week (not alphabetically)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_activity = tourist_df.groupby('day_name').size().reindex(day_order)
        results['daily_activity'] = daily_activity
    
    # 3. Activity by month
    if 'month_name' in tourist_df.columns and 'month' in tourist_df.columns:
        # Sort by month number (not alphabetically)
        month_order = list(calendar.month_name)[1:]
        monthly_activity = tourist_df.groupby('month_name').size().reindex(month_order)
        results['monthly_activity'] = monthly_activity
    
    # 4. Tourist arrivals (first appearance of each tourist)
    if 'timestamp' in tourist_df.columns:
        first_appearances = tourist_df.groupby('user_id')['timestamp'].min().reset_index()
        if 'month_name' in tourist_df.columns:
            first_appearances['month'] = pd.DatetimeIndex(first_appearances['timestamp']).month
            first_appearances['month_name'] = first_appearances['month'].map(
                {i: calendar.month_name[i] for i in range(1, 13)}
            )
            tourist_arrivals = first_appearances.groupby('month_name').size().reindex(month_order)
            results['tourist_arrivals'] = tourist_arrivals
    
    # 5. Average length of stay (if data spans enough time)
    if 'timestamp' in tourist_df.columns:
        user_timespan = tourist_df.groupby('user_id')['timestamp'].agg(['min', 'max'])
        user_timespan['stay_days'] = (user_timespan['max'] - user_timespan['min']).dt.total_seconds() / (3600 * 24)
        # Filter out unreasonably long stays (likely residents)
        reasonable_stays = user_timespan[user_timespan['stay_days'] <= 30]
        results['avg_stay_days'] = reasonable_stays['stay_days'].mean()
        results['stay_distribution'] = reasonable_stays['stay_days'].describe()
    
    return results

def create_temporal_visuals(temporal_results, df, tourist_ids):
    """
    Create visualizations for temporal patterns
    """
    # Setup matplotlib parameters for better readability
    plt.rcParams.update({'font.size': 12, 'figure.figsize': (15, 10)})
    fig, axes = plt.subplots(2, 2)
    
    # 1. Activity by hour of day
    if 'hourly_activity' in temporal_results:
        hourly = temporal_results['hourly_activity']
        axes[0, 0].bar(hourly.index, hourly.values, color='skyblue')
        axes[0, 0].set_title('Tourist Activity by Hour of Day')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Number of Activities')
        axes[0, 0].set_xticks(range(0, 24, 2))
    
    # 2. Activity by day of week
    if 'daily_activity' in temporal_results:
        daily = temporal_results['daily_activity']
        axes[0, 1].bar(daily.index, daily.values, color='lightgreen')
        axes[0, 1].set_title('Tourist Activity by Day of Week')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Number of Activities')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Activity by month
    if 'monthly_activity' in temporal_results:
        monthly = temporal_results['monthly_activity']
        axes[1, 0].bar(monthly.index, monthly.values, color='salmon')
        axes[1, 0].set_title('Tourist Activity by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Number of Activities')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Tourist arrivals by month
    if 'tourist_arrivals' in temporal_results:
        arrivals = temporal_results['tourist_arrivals']
        axes[1, 1].bar(arrivals.index, arrivals.values, color='purple')
        axes[1, 1].set_title('Tourist Arrivals by Month')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Number of Tourists')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Create a time-based activity heatmap
    tourist_df = df[df['user_id'].isin(tourist_ids)].copy()
    
    if 'hour' in tourist_df.columns and 'day_of_week' in tourist_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Create a cross-tabulation of day of week and hour
        heatmap_data = pd.crosstab(tourist_df['day_of_week'], tourist_df['hour'])
        
        # Reindex to ensure days are in order
        heatmap_data = heatmap_data.reindex(list(range(7)))
        
        # Replace day numbers with names
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                     4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        heatmap_data.index = [day_names[i] for i in heatmap_data.index]
        
        # Create heatmap
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='d', cbar_kws={'label': 'Number of Activities'})
        plt.title('Tourist Activity by Day and Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
    
    return fig


def create_activity_time_map(df, tourist_ids, zoom_start=8):
    """
    Create a time-based heatmap to show tourist activity patterns by hour
    """
    # Filter for tourist activities
    tourist_df = df[df['user_id'].isin(tourist_ids)].copy()
    
    # Create a map centered on Rwanda
    m = folium.Map(location=[-1.9403, 29.8739], zoom_start=zoom_start)
    
    if 'hour' in tourist_df.columns:
        # Create heat data for each hour
        hour_data = []
        for hour in range(24):
            # Filter data for this hour
            hour_df = tourist_df[tourist_df['hour'] == hour]
            
            # Create heat data
            heat_data = [[row['lat'], row['lon']] for _, row in hour_df.iterrows()]
            hour_data.append(heat_data)
        
        # Add time labels
        time_labels = [f"{hour}:00" for hour in range(24)]
        
        # Add heatmap with time slider
        HeatMapWithTime(
            hour_data,
            index=time_labels,
            auto_play=True,
        radius=15,
            max_opacity=0.8,
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
        ).add_to(m)
        
        # Add title
        title_html = '''
                 <h3 align="center" style="font-size:16px"><b>Tourist Activity by Hour</b></h3>
                 <p align="center">Use the time slider to see how tourist activity changes throughout the day</p>
                 '''
        m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def analyze_tourist_segments(df, tourist_ids):
    """
    Segment tourists based on their behavior patterns:
    1. Short-stay vs. long-stay tourists
    2. Urban vs. nature tourists
    3. High-activity vs. low-activity tourists
    """
    # Filter for tourist activities
    tourist_df = df[df['user_id'].isin(tourist_ids)].copy()
    
    # Calculate metrics for each tourist
    tourist_metrics = []
    
    for user_id, user_df in tourist_df.groupby('user_id'):
        # Skip if too few records
        if len(user_df) < 3:
            continue
        
        # 1. Calculate stay duration
        if 'timestamp' in user_df.columns:
            first_seen = user_df['timestamp'].min()
            last_seen = user_df['timestamp'].max()
            stay_duration = (last_seen - first_seen).total_seconds() / (3600 * 24)  # in days
        else:
            stay_duration = None
        
        # 2. Calculate activity intensity
        activity_count = len(user_df)
        activity_per_day = activity_count / stay_duration if stay_duration and stay_duration > 0 else None
        
        # 3. Calculate mobility (number of unique locations)
        user_df['lat_round'] = user_df['lat'].round(2)
        user_df['lon_round'] = user_df['lon'].round(2)
        user_df['location_id'] = user_df['lat_round'].astype(str) + "_" + user_df['lon_round'].astype(str)
        unique_locations = user_df['location_id'].nunique()
        
        # 4. Calculate mobility radius (distance from mean center)
        center_lat = user_df['lat'].mean()
        center_lon = user_df['lon'].mean()
        
        # Calculate max distance from center
        max_distance = 0
        for _, row in user_df.iterrows():
            # Haversine formula for distance
            R = 6371  # Earth radius in km
            dLat = math.radians(row['lat'] - center_lat)
            dLon = math.radians(row['lon'] - center_lon)
            a = math.sin(dLat/2) * math.sin(dLat/2) + \
                math.cos(math.radians(center_lat)) * math.cos(math.radians(row['lat'])) * \
                math.sin(dLon/2) * math.sin(dLon/2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            
            if distance > max_distance:
                max_distance = distance
        
        # 5. Check if visited tourist areas
        visited_areas = []
        if 'region' in user_df.columns:
            # List of known tourist areas (customize for Rwanda)
            tourist_areas = [
                'volcanoes', 'nyungwe', 'akagera', 'kivu', 'gisenyi', 
                'rubavu', 'musanze', 'national park'
            ]
            
            for area in tourist_areas:
                if user_df['region'].str.lower().str.contains(area).any():
                    visited_areas.append(area)
        
        # 6. Check urban vs. rural
        if 'region' in user_df.columns:
            urban_regions = ['kigali', 'nyamirambo', 'kimihurura', 'gisozi', 'downtown']
            urban_count = user_df['region'].str.lower().str.contains('|'.join(urban_regions)).sum()
            urban_percentage = urban_count / len(user_df) if len(user_df) > 0 else 0
        else:
            urban_percentage = None
        
        # 7. Activity timing
        if 'hour' in user_df.columns:
            day_activity = user_df[(user_df['hour'] >= 8) & (user_df['hour'] <= 18)].shape[0]
            night_activity = user_df.shape[0] - day_activity
            night_percentage = night_activity / user_df.shape[0] if user_df.shape[0] > 0 else 0
        else:
            night_percentage = None
        
        # Store metrics
        tourist_metrics.append({
            'user_id': user_id,
            'stay_duration': stay_duration,
            'activity_count': activity_count,
            'activity_per_day': activity_per_day,
            'unique_locations': unique_locations,
            'mobility_radius': max_distance,
            'visited_areas': visited_areas,
            'urban_percentage': urban_percentage,
            'night_percentage': night_percentage
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(tourist_metrics)
    
    # Create segments
    if not metrics_df.empty:
        # 1. Stay duration segment
        if 'stay_duration' in metrics_df.columns:
            metrics_df['stay_segment'] = pd.cut(
                metrics_df['stay_duration'],
                bins=[0, 3, 7, 14, float('inf')],
                labels=['Short (0-3 days)', 'Medium (4-7 days)', 'Long (8-14 days)', 'Extended (>14 days)']
            )
        
        # 2. Activity intensity segment
        if 'activity_per_day' in metrics_df.columns:
            activity_median = metrics_df['activity_per_day'].median()
            metrics_df['activity_segment'] = pd.cut(
                metrics_df['activity_per_day'],
                bins=[0, activity_median/2, activity_median, activity_median*2, float('inf')],
                labels=['Very Low', 'Low', 'Medium', 'High']
            )
        
        # 3. Mobility segment
        if 'mobility_radius' in metrics_df.columns:
            metrics_df['mobility_segment'] = pd.cut(
                metrics_df['mobility_radius'],
                bins=[0, 10, 50, 100, float('inf')],
                labels=['Local (0-10km)', 'Regional (10-50km)', 'National (50-100km)', 'Extensive (>100km)']
            )
        
        # 4. Urban vs. Rural preference
        if 'urban_percentage' in metrics_df.columns:
            metrics_df['urban_rural_segment'] = pd.cut(
                metrics_df['urban_percentage'],
                bins=[0, 0.3, 0.7, 1],
                labels=['Rural-focused', 'Mixed', 'Urban-focused']
            )
        
        # 5. Combined segmentation
        # Create main tourist segments based on behavior
        conditions = []
        choices = []
        
        if all(col in metrics_df.columns for col in ['mobility_radius', 'stay_duration', 'unique_locations']):
            conditions.append((metrics_df['mobility_radius'] > 50) & (metrics_df['stay_duration'] >= 7))
            choices.append('Country Explorer')
            
            conditions.append((metrics_df['mobility_radius'] <= 50) & (metrics_df['stay_duration'] >= 5))
            choices.append('Regional Immersion')
            
            conditions.append((metrics_df['stay_duration'] < 3) & (metrics_df['unique_locations'] <= 3))
            choices.append('Weekender')
            
            conditions.append((metrics_df['stay_duration'] < 5) & (metrics_df['unique_locations'] > 3))
            choices.append('Short Trip Multi-Stop')
            
            # Add more segments as needed...
            
            # Default category
            metrics_df['tourist_segment'] = np.select(conditions, choices, default='Other')
    
    return metrics_df

def visualize_tourist_segments(segment_df, df, tourist_ids, zoom_start=8):
    """
    Visualize the different tourist segments on a map
    """
    # Create a map centered on Rwanda
    m = folium.Map(location=[-1.9403, 29.8739], zoom_start=zoom_start)
    
    # Filter for tourists with segment data
    if segment_df is not None and not segment_df.empty and 'tourist_segment' in segment_df.columns:
        # Get unique segments
        segments = segment_df['tourist_segment'].unique()
        
        # Create a feature group for each segment
        segment_colors = {
            'Country Explorer': 'red',
            'Regional Immersion': 'blue',
            'Weekender': 'green',
            'Short Trip Multi-Stop': 'purple',
            'Other': 'gray'
        }
        
        # Create feature groups
        segment_groups = {}
        for segment in segments:
            color = segment_colors.get(segment, 'gray')
            segment_groups[segment] = folium.FeatureGroup(name=segment)
            m.add_child(segment_groups[segment])
        
        # Filter the original data for each tourist segment
        tourist_df = df[df['user_id'].isin(tourist_ids)].copy()
        
        # For each segment, plot movement patterns of a sample of tourists
        for segment in segments:
            # Get users in this segment
            segment_users = segment_df[segment_df['tourist_segment'] == segment]['user_id'].tolist()
            
            # Sample users (max 5 per segment)
            sample_size = min(5, len(segment_users))
            sample_users = np.random.choice(segment_users, sample_size, replace=False) if sample_size > 0 else []
            
            # Get color for this segment
            color = segment_colors.get(segment, 'gray')
            
            # For each sampled user
            for user_id in sample_users:
                # Get user data sorted by time
                user_df = tourist_df[tourist_df['user_id'] == user_id].sort_values('timestamp')
                
                # Skip if too few points
                if len(user_df) < 3:
                    continue
                
                # Get coordinates for path
                locations = user_df[['lat', 'lon']].values.tolist()
                
                # Add path
                folium.PolyLine(
                    locations=locations,
                    color=color,
                    weight=3,
                    opacity=0.7,
                    popup=f"Tourist Type: {segment}<br>User: {user_id}"
                ).add_to(segment_groups[segment])
                
                # Add markers for significant points
                folium.CircleMarker(
                    location=locations[0],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    popup=f"Start: {user_df['timestamp'].iloc[0]}"
                ).add_to(segment_groups[segment])
                
                folium.CircleMarker(
                    location=locations[-1],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    popup=f"End: {user_df['timestamp'].iloc[-1]}"
                ).add_to(segment_groups[segment])
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add title
    title_html = '''
             <h3 align="center" style="font-size:16px"><b>Tourist Segments in Rwanda</b></h3>
             <p align="center">Toggle layers to see movement patterns of different tourist types</p>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def create_tourist_dashboard(results_dict):
    """
    Create a comprehensive HTML dashboard with the analysis results
    """
    # Create an HTML template
    html_template = html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rwanda Tourism Mobility Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #4CAF50;
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
            }}
            .section {{
                background-color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .map-container {{
                height: 500px;
                margin-bottom: 20px;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .stats {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .stat-box {{
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 15px;
                width: calc(25% - 20px);
                box-sizing: border-box;
            }}
            .stat-number {{
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
            }}
            .stat-label {{
                font-size: 14px;
                color: #666;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Rwanda Tourism Mobility Analysis</h1>
            <p>Based on CDR Data Analysis</p>
        </div>
        
        <div class="container">
            <div class="section">
                <h2>Tourism Overview</h2>
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-number">{tourist_count}</div>
                        <div class="stat-label">Identified Tourists</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{hotspot_count}</div>
                        <div class="stat-label">Tourist Hotspots</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{avg_stay:.1f}</div>
                        <div class="stat-label">Avg. Stay (days)</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{peak_month}</div>
                        <div class="stat-label">Peak Month</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Tourist Hotspots</h2>
                <div class="map-container">
                    {hotspot_map}
                </div>
                <h3>Top Tourist Locations</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Region</th>
                        <th>Visitors</th>
                        <th>Avg. Time Spent (hrs)</th>
                        <th>Popular Times</th>
                    </tr>
                    {hotspot_table_rows}
                </table>
            </div>
            
            <!-- Rest of the HTML template... -->
        </div>
    </body>
    </html>
    """
    
    # Fill in the template with actual data
    if 'tourist_count' in results_dict:
        tourist_count = results_dict['tourist_count']
    else:
        tourist_count = "N/A"
    
    if 'hotspots_df' in results_dict and not results_dict['hotspots_df'].empty:
        hotspot_count = len(results_dict['hotspots_df'])
        
        # Generate hotspot table rows
        hotspot_rows = ""
        for i, hotspot in results_dict['hotspots_df'].head(10).iterrows():
            popular_times = ", ".join([f"{h}:00" for h in hotspot['busy_hours'][:2]]) if 'busy_hours' in hotspot else "N/A"
            hotspot_rows += f"""
            <tr>
                <td>{i+1}</td>
                <td>{hotspot['region']}</td>
                <td>{hotspot['unique_visitors']}</td>
                <td>{hotspot['avg_time_spent']:.2f}</td>
                <td>{popular_times}</td>
            </tr>
            """
    else:
        hotspot_count = "N/A"
        hotspot_rows = "<tr><td colspan='5'>No hotspot data available</td></tr>"
    
    if 'temporal_results' in results_dict:
        if 'avg_stay_days' in results_dict['temporal_results']:
            avg_stay = results_dict['temporal_results']['avg_stay_days']
        else:
            avg_stay = 0
        
        if 'monthly_activity' in results_dict['temporal_results']:
            monthly = results_dict['temporal_results']['monthly_activity']
            peak_month = monthly.idxmax() if not monthly.empty else "N/A"
        else:
            peak_month = "N/A"
        
        if 'hourly_activity' in results_dict['temporal_results']:
            hourly = results_dict['temporal_results']['hourly_activity']
            peak_hour = f"{hourly.idxmax()}:00" if not hourly.empty else "N/A"
        else:
            peak_hour = "N/A"
        
        if 'daily_activity' in results_dict['temporal_results']:
            daily = results_dict['temporal_results']['daily_activity']
            peak_day = daily.idxmax() if not daily.empty else "N/A"
        else:
            peak_day = "N/A"
        
        if 'tourist_arrivals' in results_dict['temporal_results']:
            arrivals = results_dict['temporal_results']['tourist_arrivals']
            arrivals_peak = arrivals.idxmax() if not arrivals.empty else "N/A"
        else:
            arrivals_peak = "N/A"
        
        if 'stay_distribution' in results_dict['temporal_results']:
            stay_stats = results_dict['temporal_results']['stay_distribution']
            stay_mode = stay_stats['50%']  # median as approximation of mode
        else:
            stay_mode = 0
    else:
        avg_stay = 0
        peak_month = "N/A"
        peak_hour = "N/A"
        peak_day = "N/A"
        arrivals_peak = "N/A"
        stay_mode = 0
    
    # Region flow table
    if 'region_flows' in results_dict and results_dict['region_flows'] is not None:
        flow_rows = ""
        for _, flow in results_dict['region_flows'].head(10).iterrows():
            flow_rows += f"""
            <tr>
                <td>{flow['from_region']}</td>
                <td>{flow['to_region']}</td>
                <td>{flow['flow_count']}</td>
            </tr>
            """
    else:
        flow_rows = "<tr><td colspan='3'>No flow data available</td></tr>"
    
    # Segment table
    if 'segment_df' in results_dict and not results_dict['segment_df'].empty:
        segment_counts = results_dict['segment_df']['tourist_segment'].value_counts()
        
        segment_rows = ""
        for segment, count in segment_counts.items():
            segment_data = results_dict['segment_df'][results_dict['segment_df']['tourist_segment'] == segment]
            avg_stay = segment_data['stay_duration'].mean() if 'stay_duration' in segment_data else 0
            avg_radius = segment_data['mobility_radius'].mean() if 'mobility_radius' in segment_data else 0
            
            # Characteristics based on segment type
            if segment == 'Country Explorer':
                characteristics = "Long stays, extensive travel across multiple regions"
            elif segment == 'Regional Immersion':
                characteristics = "Medium-length stays focused in one region"
            elif segment == 'Weekender':
                characteristics = "Short stays with limited movement"
            elif segment == 'Short Trip Multi-Stop':
                characteristics = "Brief visits to multiple locations"
            else:
                characteristics = "Mixed patterns"
            
            segment_rows += f"""
            <tr>
                <td>{segment}</td>
                <td>{count}</td>
                <td>{avg_stay:.1f}</td>
                <td>{avg_radius:.1f}</td>
                <td>{characteristics}</td>
            </tr>
            """
    else:
        segment_rows = "<tr><td colspan='5'>No segment data available</td></tr>"
    
    # Maps
    hotspot_map = '<iframe src="tourist_hotspots.html" width="100%" height="100%"></iframe>'
    flow_map = '<iframe src="tourist_flows.html" width="100%" height="100%"></iframe>'
    time_map = '<iframe src="activity_time.html" width="100%" height="100%"></iframe>'
    segment_map = '<iframe src="tourist_segments.html" width="100%" height="100%"></iframe>'
    
    # Fill in the template
    dashboard_html = html_template.format(
        tourist_count=tourist_count,
        hotspot_count=hotspot_count,
        avg_stay=avg_stay,
        peak_month=peak_month,
        hotspot_map=hotspot_map,
        hotspot_table_rows=hotspot_rows,
        flow_map=flow_map,
        flow_table_rows=flow_rows,
        time_map=time_map,
        peak_hour=peak_hour,
        peak_day=peak_day,
        arrivals_peak=arrivals_peak,
        stay_mode=stay_mode,
        segment_map=segment_map,
        segment_table_rows=segment_rows
    )
    
    return dashboard_html


def run_tourism_mobility_analysis(cdr_file_path='cdr_data.csv'):
    """
    Main function to run the complete tourism mobility analysis
    """
    print("Starting Rwanda Tourism Mobility Analysis...")
    results = {}
    
    # 1. Load data
    df = load_real_cdr_data(cdr_file_path)
    if df is None:
        print("Failed to load data. Exiting analysis.")
        return None
    
    # 2. Identify tourists
    tourist_ids = identify_tourists(df)
    results['tourist_count'] = len(tourist_ids)
    
    # 3. Create tourist heatmap
    tourist_heatmap = create_tourist_heatmap(df, tourist_ids)
    tourist_heatmap.save('tourist_heatmap.html')
    print("Saved tourist heatmap to tourist_heatmap.html")
    
    # 4. Analyze tourist hotspots
    hotspots_df, tourist_df_with_clusters = analyze_tourist_hotspots(df, tourist_ids)
    results['hotspots_df'] = hotspots_df
    
    hotspot_map = visualize_tourist_hotspots(hotspots_df, tourist_df_with_clusters)
    hotspot_map.save('tourist_hotspots.html')
    print("Saved tourist hotspots map to tourist_hotspots.html")
    
    # 5. Analyze tourist movement patterns
    location_flows, region_flows, transitions = analyze_tourist_movement_patterns(df, tourist_ids)
    results['region_flows'] = region_flows
    
    flow_map = visualize_tourist_flows(location_flows, region_flows)
    flow_map.save('tourist_flows.html')
    print("Saved tourist flow map to tourist_flows.html")
    
    # 6. Analyze temporal patterns
    temporal_results = analyze_temporal_patterns(df, tourist_ids)
    results['temporal_results'] = temporal_results
    
    time_map = create_activity_time_map(df, tourist_ids)
    time_map.save('activity_time.html')
    print("Saved activity time map to activity_time.html")
    
    # 7. Create temporal visualizations
    temporal_fig = create_temporal_visuals(temporal_results, df, tourist_ids)
    temporal_fig.savefig('temporal_patterns.png')
    print("Saved temporal patterns chart to temporal_patterns.png")
    
    # 8. Segment tourists
    segment_df = analyze_tourist_segments(df, tourist_ids)
    results['segment_df'] = segment_df
    
    segment_map = visualize_tourist_segments(segment_df, df, tourist_ids)
    segment_map.save('tourist_segments.html')
    print("Saved tourist segment map to tourist_segments.html")
    
    # 9. Create dashboard
    dashboard_html = create_tourist_dashboard(results)
    with open('tourism_dashboard.html', 'w') as f:
        f.write(dashboard_html)
    print("Saved comprehensive dashboard to tourism_dashboard.html")
    
    print("\nAnalysis complete! All results have been saved.")
    
    return results

# Example usage
if __name__ == "__main__":
    run_tourism_mobility_analysis('/Users/brightabohsilasedem/Desktop/geospatial/rwanda_mobile_data.csv')