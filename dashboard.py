import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Lebanon Cell Tower Infrastructure",
    # page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data(filepath):
    """Load cleaned cell tower data"""
    df = pd.read_csv(filepath)
    return df

# ============================================================================
# MAIN APP
# ============================================================================

# Title
st.title("Lebanon Cell Tower Infrastructure Dashboard")
st.markdown("### Comprehensive Analysis of Telecommunications Network")
st.markdown("---")

# Load data
try:
    df = load_data(r"data/cell_towers_lebanon.csv")
except FileNotFoundError:
    st.error("❌ Data file not found! Please check the file path.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Display data info
st.sidebar.markdown(f"**Total Records:** {len(df):,}")
st.sidebar.markdown(f"**Columns:** {len(df.columns)}")

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================
# st.sidebar.header("🔍 Filters")
st.sidebar.header("Filters")

# Dynamically detect available columns
available_columns = df.columns.tolist()

# provider filter (try different possible column names)
provider_col = 'provider'

if provider_col:
    providers = ['All'] + sorted(df[provider_col].dropna().unique().tolist())
    selected_providers = st.sidebar.multiselect(
        "Provider",
        options=providers,
        default=['All']
    )
else:
    selected_providers = ['All']
    st.sidebar.warning("No provider column found")

# Technology filter
tech_col = 'radio'

if tech_col:
    technologies = ['All'] + sorted(df[tech_col].dropna().unique().tolist())
    selected_technologies = st.sidebar.multiselect(
        "Technology",
        options=technologies,
        default=['All']
    )
else:
    selected_technologies = ['All']
    st.sidebar.warning("No technology column found")

# Governorate filter
gov_col = 'governorate'

if gov_col:
    governorates = ['All'] + sorted(df[gov_col].dropna().unique().tolist())
    selected_governorates = st.sidebar.multiselect(
        "Governorate",
        options=governorates,
        default=['All']
    )
else:
    selected_governorates = ['All']

# District filter
dist_col = 'district'

if dist_col:
    districts = ['All'] + sorted(df[dist_col].dropna().unique().tolist())
    selected_districts = st.sidebar.multiselect(
        "District",
        options=districts,
        default=['All']
    )
else:
    selected_districts = ['All']

# Signal range filter
range_col = 'range'

if range_col:
    min_range = float(df[range_col].min())
    max_range = float(df[range_col].max())
    
    range_filter = st.sidebar.slider(
        "Signal Range (meters)",
        min_value=int(min_range),
        max_value=int(max_range),
        value=(int(min_range), int(max_range))
    )
else:
    range_filter = None

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard visualizes cell tower infrastructure in Lebanon. "
    "Data includes location, provider, technology, and coverage information."
)

# ============================================================================
# APPLY FILTERS
# ============================================================================
filtered_df = df.copy()

# provider filter
if provider_col and 'All' not in selected_providers and len(selected_providers) > 0:
    filtered_df = filtered_df[filtered_df[provider_col].isin(selected_providers)]

# Technology filter
if tech_col and 'All' not in selected_technologies and len(selected_technologies) > 0:
    filtered_df = filtered_df[filtered_df[tech_col].isin(selected_technologies)]

# Governorate filter
if gov_col and 'All' not in selected_governorates and len(selected_governorates) > 0:
    filtered_df = filtered_df[filtered_df[gov_col].isin(selected_governorates)]

# District filter
if dist_col and 'All' not in selected_districts and len(selected_districts) > 0:
    filtered_df = filtered_df[filtered_df[dist_col].isin(selected_districts)]

# Range filter
if range_col and range_filter:
    filtered_df = filtered_df[
        (filtered_df[range_col] >= range_filter[0]) &
        (filtered_df[range_col] <= range_filter[1])
    ]

# ============================================================================
# KPIs
# ============================================================================
# st.markdown("## 📊 Key Metrics")
st.markdown("## Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_towers = len(filtered_df)
    st.metric(
        label="Total Towers",
        value=f"{total_towers:,}",
        delta=f"{(total_towers/len(df)*100):.1f}% of total" if len(df) > 0 else None
    )

with col2:
    if tech_col:
        # Try to identify 4G
        towers_4g = len(filtered_df[filtered_df[tech_col].str.contains('4G|LTE', case=False, na=False)])
        pct_4g = (towers_4g / total_towers * 100) if total_towers > 0 else 0
        st.metric(
            label="4G Towers",
            value=f"{towers_4g:,}",
            delta=f"{pct_4g:.1f}%"
        )
    else:
        st.metric(label="4G Towers", value="N/A")

with col3:
    if tech_col:
        # Try to identify 3G
        towers_3g = len(filtered_df[filtered_df[tech_col].str.contains('3G|UMTS', case=False, na=False)])
        pct_3g = (towers_3g / total_towers * 100) if total_towers > 0 else 0
        st.metric(
            label="3G Towers",
            value=f"{towers_3g:,}",
            delta=f"{pct_3g:.1f}%"
        )
    else:
        st.metric(label="3G Towers", value="N/A")

with col4:
    if range_col:
        avg_range = filtered_df[range_col].mean() / 1000  # Convert to km
        st.metric(
            label="Avg Signal Range",
            value=f"{avg_range:.2f} km"
        )
    else:
        st.metric(label="Avg Signal Range", value="N/A")

with col5:
    if provider_col:
        providers_count = filtered_df[provider_col].nunique()
        st.metric(
            label="Providers",
            value=providers_count
        )
    else:
        st.metric(label="Providers", value="N/A")

st.markdown("---")

# ============================================================================
# CHARTS SECTION
# ============================================================================
#st.markdown("## 📈 Network Analysis")
st.markdown("## Network Analysis")

# Two columns for charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if tech_col:
        st.markdown("### Technology Distribution")
        
        tech_counts = filtered_df[tech_col].value_counts()
        
        fig_tech = px.pie(
            values=tech_counts.values,
            names=tech_counts.index,
            hole=0.4
        )
        fig_tech.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        fig_tech.update_layout(
            showlegend=True,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_tech, use_container_width=True)
    else:
        st.info("Technology column not found")

with chart_col2:
    if provider_col:
        st.markdown("### Provider Distribution")
        
        op_counts = filtered_df[provider_col].value_counts()
        
        fig_op = px.bar(
            x=op_counts.index,
            y=op_counts.values,
            labels={'x': 'provider', 'y': 'Number of Towers'}
        )
        fig_op.update_traces(
            hovertemplate='<b>%{x}</b><br>Towers: %{y}<extra></extra>'
        )
        fig_op.update_layout(
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="provider",
            yaxis_title="Number of Towers"
        )
        st.plotly_chart(fig_op, use_container_width=True)
    else:
        st.info("provider column not found")

# ============================================================================
# SECOND ROW OF CHARTS
# ============================================================================
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    if provider_col and tech_col:
        st.markdown("### Towers by Provider & Technology")
        
        cross_tab = pd.crosstab(filtered_df[provider_col], filtered_df[tech_col])
        
        fig_stack = go.Figure()
        
        for tech in cross_tab.columns:
            fig_stack.add_trace(go.Bar(
                name=tech,
                x=cross_tab.index,
                y=cross_tab[tech],
                hovertemplate='<b>%{x}</b><br>' + tech + ': %{y}<extra></extra>'
            ))
        
        fig_stack.update_layout(
            barmode='stack',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="provider",
            yaxis_title="Number of Towers",
            legend_title="Technology"
        )
        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("provider or Technology column not found")

with chart_col4:
    if gov_col and filtered_df[gov_col].notna().any():
        st.markdown("### Top 10 Areas by Tower Count")
        
        gov_counts = filtered_df[gov_col].value_counts().head(10)
        
        fig_gov = px.bar(
            x=gov_counts.values,
            y=gov_counts.index,
            orientation='h',
            labels={'x': 'Number of Towers', 'y': 'Area'}
        )
        fig_gov.update_traces(
            hovertemplate='<b>%{y}</b><br>Towers: %{x}<extra></extra>'
        )
        fig_gov.update_layout(
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gov, use_container_width=True)
    
    elif range_col and filtered_df[range_col].notna().any():
        st.markdown("### Signal Range Distribution")
        
        fig_hist = px.histogram(
            filtered_df[filtered_df[range_col].notna()],
            x=range_col,
            nbins=50,
            labels={range_col: 'Signal Range (meters)'}
        )
        fig_hist.update_traces(
            hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
        )
        fig_hist.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Signal Range (meters)",
            yaxis_title="Number of Towers"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No additional data available for visualization")

st.markdown("---")

# ============================================================================
# MAP SECTION
# ============================================================================
# st.markdown("## 🗺️ Tower Location Map")
st.markdown("## Tower Location Map")

# Find latitude and longitude columns
lat_col = 'lat'
lon_col = 'lon'

if not lat_col or not lon_col:
    # st.error("❌ Could not find latitude/longitude columns in the data")
    st.error("Could not find latitude/longitude columns in the data")
else:
    # Map controls
    col_map1, col_map2, col_map3, col_map4 = st.columns([2, 2, 2, 2])
    
    with col_map1:
        map_type = st.selectbox(
            "Map Type:",
            options=["Interactive Map (PyDeck)", "Detailed Map (Folium)", "Heatmap"]
        )
    
    with col_map2:
        basemap_style = st.selectbox(
            "Basemap:",
            options=[
                "OpenStreetMap",
                "Satellite",
                "Terrain",
                "Dark",
                "Light"
            ]
        )
    
    
    with col_map4:
        show_lebanon = st.checkbox("Show Lebanon Border", value=True)
        show_governorates = st.checkbox("Show Governorates", value=False)
    
    st.markdown("---")
    
    # Cache function to load boundaries
    @st.cache_data
    def load_boundaries():
        import geopandas as gpd
    
        lebanon_path = "data/Lebanon.geojson"
        gov_path = "data/Governorates.geojson"
    
        try:
            lebanon_gdf = gpd.read_file(lebanon_path)
            if lebanon_gdf.crs != 'EPSG:4326':
                lebanon_gdf = lebanon_gdf.to_crs('EPSG:4326')
            lebanon_json = lebanon_gdf.__geo_interface__
        except Exception as e:
            st.warning(f"Could not load Lebanon boundary: {e}")
            lebanon_json = None
    
        try:
            gov_gdf = gpd.read_file(gov_path)
            if gov_gdf.crs != 'EPSG:4326':
                gov_gdf = gov_gdf.to_crs('EPSG:4326')
            gov_json = gov_gdf.__geo_interface__
        except Exception as e:
            st.warning(f"Could not load Governorates: {e}")
            gov_json = None
    
        return lebanon_json, gov_json


    if map_type == "Interactive Map (PyDeck)":
        
        # Prepare data for PyDeck
        map_data = filtered_df[[lon_col, lat_col]].copy()
        map_data.columns = ['longitude', 'latitude']
        
        # Add color if provider exists
        if provider_col:
            map_data['provider'] = filtered_df[provider_col].values
            
            # Color mapping for providers (matching Folium colors)
            unique_providers = map_data['provider'].unique()
            colors = [
                [255, 107, 107, 200],  # Red
                [78, 205, 196, 200],   # Turquoise
                [255, 159, 64, 200],   # Orange
                [153, 102, 255, 200],  # Purple
                [149, 165, 166, 200]   # Gray
            ]
            
            color_map = {op: colors[i % len(colors)] for i, op in enumerate(unique_providers)}
            map_data['color'] = map_data['provider'].map(color_map)
        else:
            map_data['color'] = [[78, 205, 196, 200]] * len(map_data)
        
        # Add technology for tooltip if available
        if tech_col:
            map_data['technology'] = filtered_df[tech_col].values
        
        # Basemap style mapping for PyDeck (using publicly available styles)
        basemap_styles = {
            "OpenStreetMap": "road",
            "Satellite": "satellite",
            "Terrain": "outdoors",
            "Dark": "dark",
            "Light": "light"
        }
        
        # Layers list
        layers = []
        
        # Tower points layer
        tower_layer = pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position='[longitude, latitude]',
            get_color='color',
            get_radius=300,
            pickable=True,
            auto_highlight=True
        )
        
        layers.append(tower_layer)
        
        # View state
        view_state = pdk.ViewState(
            latitude=map_data['latitude'].mean(),
            longitude=map_data['longitude'].mean(),
            zoom=8,
            pitch=0
        )
        
        # Tooltip
        tooltip_html = "<b>Tower Information</b><br/>"
        if provider_col:
            tooltip_html += "<b>Provider:</b> {provider}<br/>"
        if tech_col:
            tooltip_html += "<b>Technology:</b> {technology}<br/>"
        
        tooltip = {
            "html": tooltip_html,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
        
        # Render with simplified map_style
        r = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style=basemap_styles.get(basemap_style, "road")
        )
        
        # Display map
        st.pydeck_chart(r)
        
        # Create legend overlay if provider column exists
        if provider_col:
            # Color hex mapping (matching Folium colors)
            color_hex_map = {
                unique_providers[i]: [
                    '#FF6B6B',  # Red
                    '#4ECDC4',  # Turquoise
                    '#FF9F40',  # Orange
                    '#9966FF',  # Purple
                    '#95A5A6'   # Gray
                ][i % 5]
                for i in range(len(unique_providers))
            }
            
            # Build legend items HTML (matching Folium legend structure)
            legend_items_html = ""
            for provider in unique_providers:
                color_hex = color_hex_map[provider]
                legend_items_html += f'<p style="margin: 5px 0;"><span style="background-color: {color_hex}; width: 15px; height: 15px; display: inline-block; border-radius: 50%; margin-right: 5px;"></span>{provider}</p>'
            
            # Display legend with negative margin to overlay on map (matching Folium legend style)
            st.markdown(
                f'<div style="margin-top: -150px; margin-right: 8px; float: right; background-color: white; z-index: 1000; border: 2px solid grey; border-radius: 5px; padding: 10px; width: 200px; height: auto; font-size: 14px;"><p style="margin: 0 0 10px 0; font-weight: bold;">Provider Legend</p>{legend_items_html}</div><div style="clear: both;"></div>',
                unsafe_allow_html=True
            )
        
        # st.info("💡 Note: Interactive map uses simplified basemaps. For detailed basemaps, use 'Detailed Map (Folium)'")
        st.info("Note: Interactive map uses simplified basemaps. For detailed basemaps, use 'Detailed Map (Folium)'")

    elif map_type == "Detailed Map (Folium)":
        
        with st.spinner("Loading map..."):
            # Basemap tiles for Folium
            basemap_tiles = {
                "OpenStreetMap": "OpenStreetMap",
                "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "Terrain": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
                "Dark": "CartoDB dark_matter",
                "Light": "CartoDB positron"
            }
            
            # Create folium map
            center_lat = filtered_df[lat_col].mean()
            center_lon = filtered_df[lon_col].mean()
            
            if basemap_style in ["OpenStreetMap", "Dark", "Light"]:
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=9,
                    tiles=basemap_tiles[basemap_style],
                    prefer_canvas=True
                )
            else:
                # For custom tile servers
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=9,
                    prefer_canvas=True
                )
                folium.TileLayer(
                    tiles=basemap_tiles[basemap_style],
                    attr=basemap_style,
                    name=basemap_style
                ).add_to(m)
            
            # Load boundaries if needed
            if show_lebanon or show_governorates:
                lebanon_json, gov_json = load_boundaries()
                
                # Add Lebanon boundary layer
                if show_lebanon and lebanon_json:
                    folium.GeoJson(
                        lebanon_json,
                        name='Lebanon Border',
                        style_function=lambda x: {
                            'fillColor': 'none',
                            'color': 'black',
                            'weight': 2,
                            'fillOpacity': 0
                        }
                    ).add_to(m)
                
                # Add Governorates layer
                if show_governorates and gov_json:
                    folium.GeoJson(
                        gov_json,
                        name='Governorates',
                        style_function=lambda x: {
                            'fillColor': 'none',
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0
                        }
                    ).add_to(m)
            
            # Color mapping for providers
            if provider_col:
                unique_providers = filtered_df[provider_col].unique()
                provider_colors = {
                    unique_providers[i]: [
                    '#FF6B6B',  # Red
                    '#4ECDC4',  # Turquoise
                    '#FF9F40',  # Orange
                    '#9966FF',  # Purple
                    '#95A5A6'   # Gray
                ][i % 5]
                    for i in range(len(unique_providers))
                }
            
            # Add marker cluster
            marker_cluster = MarkerCluster(
                options={
                    'maxClusterRadius': 50,
                    'disableClusteringAtZoom': 13
                }
            ).add_to(m)
            
            # Add markers (limit to avoid performance issues)
            sample_size = min(500, len(filtered_df))  # Reduced from 1000
            sample_df = filtered_df.sample(sample_size) if len(filtered_df) > sample_size else filtered_df
            
            for idx, row in sample_df.iterrows():
                popup_text = f"<b>Location:</b> ({row[lat_col]:.4f}, {row[lon_col]:.4f})<br>"
                
                if provider_col:
                    popup_text += f"<b>Provider:</b> {row[provider_col]}<br>"
                    marker_color = provider_colors.get(row[provider_col], 'gray')
                else:
                    marker_color = 'blue'
                
                if tech_col:
                    popup_text += f"<b>Technology:</b> {row[tech_col]}<br>"
                if range_col:
                    popup_text += f"<b>Range:</b> {row[range_col]}m"
                
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=4,
                    popup=folium.Popup(popup_text, max_width=200),
                    color=marker_color,
                    fill=True,
                    fillOpacity=0.7
                ).add_to(marker_cluster)
            
            # Add HTML legend to the map
            if provider_col:
                legend_html = '''
                <div style="position: fixed; 
                            bottom: 25px; right: 50px; width: 200px; height: auto; 
                            background-color: white; z-index:9999; font-size:14px;
                            border:2px solid grey; border-radius: 5px; padding: 10px">
                <p style="margin: 0 0 10px 0; font-weight: bold;">Provider Legend</p>
                '''
                
                for provider, color in provider_colors.items():
                    legend_html += f'''
                    <p style="margin: 5px 0;">
                        <span style="background-color: {color}; width: 15px; height: 15px; 
                                     display: inline-block; border-radius: 50%; margin-right: 5px;"></span>
                        {provider}
                    </p>
                    '''
                
                legend_html += '</div>'
                m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add layer control
            folium.LayerControl().add_to(m)
        
        # Display
        st_folium(m, width=1200, height=600, returned_objects=[])
        
        if len(filtered_df) > sample_size:
            st.info(f"Showing {sample_size} random towers out of {len(filtered_df):,} for performance")

    else:  # Heatmap
        
        with st.spinner("Generating heatmap..."):
            # Basemap tiles for Folium
            basemap_tiles = {
                "OpenStreetMap": "OpenStreetMap",
                "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "Terrain": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
                "Dark": "CartoDB dark_matter",
                "Light": "CartoDB positron"
            }
            
            # Create heatmap
            center_lat = filtered_df[lat_col].mean()
            center_lon = filtered_df[lon_col].mean()
            
            if basemap_style in ["OpenStreetMap", "Dark", "Light"]:
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=9,
                    tiles=basemap_tiles[basemap_style],
                    prefer_canvas=True
                )
            else:
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=9,
                    prefer_canvas=True
                )
                folium.TileLayer(
                    tiles=basemap_tiles[basemap_style],
                    attr=basemap_style,
                    name=basemap_style
                ).add_to(m)
            
            # Load boundaries if needed
            if show_lebanon or show_governorates:
                lebanon_json, gov_json = load_boundaries()
                
                # Add Lebanon boundary layer
                if show_lebanon and lebanon_json:
                    folium.GeoJson(
                        lebanon_json,
                        name='Lebanon Border',
                        style_function=lambda x: {
                            'fillColor': 'none',
                            'color': 'black',
                            'weight': 2,
                            'fillOpacity': 0
                        }
                    ).add_to(m)
                
                # Add Governorates layer
                if show_governorates and gov_json:
                    folium.GeoJson(
                        gov_json,
                        name='Governorates',
                        style_function=lambda x: {
                            'fillColor': 'none',
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0
                        }
                    ).add_to(m)
            
            # Prepare data for heatmap
            heat_data = [[row[lat_col], row[lon_col]] for idx, row in filtered_df.iterrows()]
            
            # Add heatmap
            HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
            
            # Add heatmap legend
            heatmap_legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; right: 50px; width: 180px; height: auto; 
                        background-color: white; z-index:9999; font-size:14px;
                        border:2px solid grey; border-radius: 5px; padding: 10px">
            <p style="margin: 0 0 10px 0; font-weight: bold;">Density Legend</p>
            <div style="background: linear-gradient(to right, blue, cyan, lime, yellow, red); 
                        height: 20px; width: 100%; border-radius: 3px;"></div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 12px;">
                <span>Low</span>
                <span>High</span>
            </div>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(heatmap_legend_html))
            
            # Add layer control
            folium.LayerControl().add_to(m)
        
        # Display
        st_folium(m, width=1200, height=600, returned_objects=[])

st.markdown("---")

# ============================================================================
# DATA TABLE
# ============================================================================
#st.markdown("## 📋 Tower Data Table")
st.markdown("## Tower Data Table")

# Show/hide toggle
show_data = st.checkbox("Show detailed data table", value=False)

if show_data:
    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        height=400
    )
    
    if len(filtered_df) > 100:
        st.info(f"Showing first 100 rows out of {len(filtered_df)} towers")

# ============================================================================
# EXPORT SECTION
# ============================================================================
st.markdown("---")
# st.markdown("## 💾 Export Data")
st.markdown("## Export Data")

col_export1, col_export2, col_export3 = st.columns(3)

with col_export1:
    # if st.button("📥 Export Filtered Data (CSV)"):
    if st.button("Export Filtered Data (CSV)"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="cell_towers_filtered.csv",
            mime="text/csv"
        )

with col_export2:
    # if st.button("📥 Export Summary Statistics"):
    if st.button("Export Summary Statistics"):
        summary_stats = {
            'Total Towers': len(filtered_df),
            'Filters Applied': {
                'providers': selected_providers if provider_col else 'N/A',
                'Technologies': selected_technologies if tech_col else 'N/A',
                'Governorates': selected_governorates if gov_col else 'N/A',
                'Districts': selected_districts if dist_col else 'N/A'
            }
        }
        
        if provider_col:
            summary_stats['By provider'] = filtered_df[provider_col].value_counts().to_dict()
        if tech_col:
            summary_stats['By Technology'] = filtered_df[tech_col].value_counts().to_dict()
        
        import json
        json_str = json.dumps(summary_stats, indent=2)
        
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="summary_statistics.json",
            mime="application/json"
        )

with col_export3:
    #st.info("💡 Use filters in sidebar to customize your data export")
    st.info("Use filters in sidebar to customize your data export")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Lebanon Cell Tower Infrastructure Dashboard</p>
        <p>Data Source: OpenCelliD | Last Updated: 2026</p>
        <p>Built with Streamlit & Plotly</p>
    </div>

""", unsafe_allow_html=True)


