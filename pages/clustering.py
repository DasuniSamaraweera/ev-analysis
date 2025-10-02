import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pydeck as pdk

def show_clustering_page(df):
    st.header("🌍 EV Clustering & Infrastructure Planning")
    st.write("Cluster EVs by electric range and location to guide charging station placement.")

    # -----------------------------
    # Preprocessing
    # -----------------------------
    if "Vehicle Location" not in df.columns or "Electric Range" not in df.columns:
        st.warning("Dataset missing required columns (Electric Range, Vehicle Location).")
        return

    # Keep only needed columns
    df = df[['Electric Range', 'Vehicle Location']].dropna()

    # Extract latitude & longitude from "POINT (long lat)" format
    df['Longitude'] = df['Vehicle Location'].str.extract(r'POINT \((-?\d+\.\d+)')
    df['Latitude'] = df['Vehicle Location'].str.extract(r' (-?\d+\.\d+)\)')
    df[['Longitude', 'Latitude']] = df[['Longitude', 'Latitude']].astype(float)

    # Final clean dataset
    df = df[['Electric Range', 'Latitude', 'Longitude']].dropna()
    
    # ✅ FIXED: Validate coordinate bounds to prevent map errors
    df = df[
        (df['Latitude'].between(-90, 90)) &
        (df['Longitude'].between(-180, 180)) &
        (df['Electric Range'] >= 0)
    ]
    
    if len(df) == 0:
        st.error("No valid data found after cleaning. Please check the dataset.")
        return

    st.subheader("Processed Data Sample")
    st.write(df.head())

    # -----------------------------
    # Clustering
    # -----------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['Electric Range', 'Latitude', 'Longitude']])

    st.sidebar.header("Clustering Options")
    k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)

    st.subheader("Clustered Data")
    st.write(df.head())

    # -----------------------------
    # Visualization - Scatter Plot
    # -----------------------------
    st.subheader("📊 Scatter Plot of Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='tab10', alpha=0.6)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("EV Clusters by Location & Range")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)

    # -----------------------------
    # Visualization - Interactive Map
    # -----------------------------
    st.subheader("🌍 Interactive Map of EV Clusters")

    # ✅ FIXED: Additional validation before creating map
    if len(df) == 0:
        st.error("No data available for mapping after clustering.")
        return
        
    # Ensure coordinates are reasonable for mapping
    lat_center = df['Latitude'].median()
    lon_center = df['Longitude'].median()
    
    if not (-90 <= lat_center <= 90) or not (-180 <= lon_center <= 180):
        st.error("Invalid coordinate data detected. Cannot create map.")
        return

    cluster_colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [255, 165, 0], [128, 0, 128], [0, 128, 128], [128, 128, 0]
    ]

    df['color'] = df['Cluster'].apply(lambda x: cluster_colors[x % len(cluster_colors)])

    # ✅ FIXED: Limit data size to prevent memory issues
    map_df = df.head(5000)  # Limit to 5000 points max
    
    if len(df) > 5000:
        st.info(f"Showing {len(map_df)} out of {len(df)} points on the map for performance.")

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[Longitude, Latitude]',
        get_radius=200,
        get_color='color',
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=lat_center,
        longitude=lon_center,
        zoom=7,
        pitch=0,
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Cluster: {Cluster}\nRange: {Electric Range}"}
    )
    
    try:
        st.pydeck_chart(r)
    except Exception as e:
        st.error(f"Map rendering failed: {str(e)}")
        st.info("Try reducing the number of clusters or check your data for invalid coordinates.")
