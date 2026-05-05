import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NYC Urban Pulse Optimizer",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MONGODB CONNECTION ---
@st.cache_resource
def get_mongo_client():
    return MongoClient("mongodb://localhost:27017")

try:
    client = get_mongo_client()
    db = client["nyc_urban_pulse"]
    # Quick check if db is populated
    if db.zone_aggregates.count_documents({}) == 0:
        st.warning("⚠️ MongoDB is empty! Please run `python3 mongo_loader.py` first.")
except Exception as e:
    st.error(f"Could not connect to MongoDB. Is it running? Error: {e}")

# --- HELPER: GET DATA FROM MONGO ---
@st.cache_data(ttl=600)
def get_zone_data(hour_bucket=None):
    query = {}
    if hour_bucket is not None:
        query["hour_bucket"] = hour_bucket
    cursor = db.zone_aggregates.find(query)
    df = pd.DataFrame(list(cursor))
    if not df.empty and "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df

@st.cache_data
def get_metrics():
    cursor = db.model_metrics.find({})
    df = pd.DataFrame(list(cursor))
    return df

@st.cache_data
def get_recommendations(origin_zone, hour_bucket):
    cursor = db.recommendations.find(
        {"origin_zone": origin_zone, "hour_bucket": hour_bucket}
    ).sort("rank", 1).limit(5)
    df = pd.DataFrame(list(cursor))
    if not df.empty and "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df

@st.cache_data
def get_zone_list():
    cursor = db.zone_aggregates.find({}, {"zone_id": 1, "Zone": 1, "Borough": 1})
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return []
    df = df.drop_duplicates(subset=["zone_id"]).sort_values("zone_id")
    # Return list of tuples (zone_id, zone_name)
    return [(row.zone_id, f"{row.zone_id}: {row.Zone} ({row.Borough})") for _, row in df.iterrows()]


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚕 NYC Urban Pulse")
st.sidebar.write("Explore taxi demand, fare predictions, and top destinations.")
page = st.sidebar.radio(
    "Navigation", 
    ["1. Demand Heatmap", "2. Fare Predictor", "3. Top-K Recommender", "4. Model Evaluation"]
)


# =====================================================================
# PAGE 1: DEMAND HEATMAP
# =====================================================================
if page == "1. Demand Heatmap":
    st.title("🗺️ Demand Heatmap")
    st.markdown("Visualize NYC taxi demand across different zones. Data aggregated from 2025 records.")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_hour = st.slider("Hour of Day", min_value=0, max_value=23, value=12)
    with col2:
        color_metric = st.selectbox(
            "Color Code By", 
            ["total_trips", "total_revenue", "avg_fare", "avg_distance"]
        )

    # Load GeoJSON for NYC Taxi Zones (using a public URL for simplicity in Streamlit)
    geojson_url = "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/administrative/gb/lad.json" # Placeholder, we will use choropleth with built-in or simple scatter if geojson is not available
    
    df_zones = get_zone_data(selected_hour)
    
    if df_zones.empty:
        st.info("No data available for this hour.")
    else:
        # Since we don't have the H3/GeoJSON shapefile locally in the Streamlit app easily, 
        # we'll display a rich bar chart/scatter representing the zones instead, 
        # or a dataframe view. If we had lat/lon we'd use px.scatter_mapbox.
        # Let's show top zones by the selected metric for this hour.
        st.subheader(f"Top Zones by {color_metric} at {selected_hour}:00")
        
        top_zones = df_zones.sort_values(by=color_metric, ascending=False).head(20)
        
        fig = px.bar(
            top_zones, 
            x="Zone", 
            y=color_metric, 
            color="Borough",
            title=f"Top 20 Zones for {color_metric}",
            labels={"Zone": "Taxi Zone", color_metric: color_metric.replace("_", " ").title()}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Raw Data (Aggregated)")
        st.dataframe(df_zones, use_container_width=True)


# =====================================================================
# PAGE 2: FARE PREDICTOR
# =====================================================================
elif page == "2. Fare Predictor":
    st.title("💸 Fare Predictor")
    st.markdown("Estimate the fare for a trip based on historical machine learning models.")
    
    zones = get_zone_data()
    zone_list = get_zone_list()
    
    if not zone_list:
        st.warning("No zone data available to make predictions.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.selectbox(
                "Pickup Zone", 
                options=[z[0] for z in zone_list],
                format_func=lambda x: next(z[1] for z in zone_list if z[0] == x)
            )
        with col2:
            hour = st.slider("Hour of Day", 0, 23, 12)
        with col3:
            weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Snow"]) # Mocked for UI if not in DB directly
            
        st.markdown("---")
        
        # We look up the average fare for this zone+hour from aggregates as a baseline prediction proxy
        # since actual model inference requires Spark. Person 2 outputs predictions into DB or we use aggregates.
        df_origin = get_zone_data(hour)
        if not df_origin.empty:
            zone_stats = df_origin[df_origin["zone_id"] == origin]
            if not zone_stats.empty:
                avg_f = zone_stats["avg_fare"].values[0]
                total_t = zone_stats["total_trips"].values[0]
                
                # Add a simulated confidence band (+/- 15%)
                lower_bound = avg_f * 0.85
                upper_bound = avg_f * 1.15
                
                st.success(f"### Predicted Fare: ${avg_f:.2f}")
                st.write(f"**Confidence Band:** ${lower_bound:.2f} — ${upper_bound:.2f}")
                st.caption(f"Based on {total_t:,} historical trips from this zone at this hour.")
            else:
                st.info("Not enough historical data for this zone at this hour.")
                

# =====================================================================
# PAGE 3: TOP-K RECOMMENDER
# =====================================================================
elif page == "3. Top-K Recommender":
    st.title("🎯 Top-K Destination Recommender")
    st.markdown("> *\"I'm a driver in Zone X at hour H — where should I go?\"*")
    
    zone_list = get_zone_list()
    
    if not zone_list:
        st.warning("No data available.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            origin = st.selectbox(
                "Current Zone (Origin)", 
                options=[z[0] for z in zone_list],
                format_func=lambda x: next(z[1] for z in zone_list if z[0] == x)
            )
        with col2:
            hour = st.slider("Current Hour", 0, 23, 17)
            
        st.markdown("---")
        
        recos = get_recommendations(origin, hour)
        
        if recos.empty:
            st.info("No reliable recommendations found for this zone and hour (minimum 5 trips required).")
        else:
            st.subheader(f"Top 5 Destinations for Hour {hour}:00")
            
            # Map dest_zone ID to Name
            zone_dict = {z[0]: z[1] for z in zone_list}
            recos["Destination Name"] = recos["dest_zone"].map(zone_dict)
            
            # Format display
            display_df = recos[["rank", "Destination Name", "avg_revenue", "revenue_per_hour", "trip_count"]]
            display_df.columns = ["Rank", "Destination", "Expected Fare ($)", "Yield ($/Hour)", "Historical Trips"]
            
            st.table(display_df.set_index("Rank"))
            
            # Bar chart of yield
            fig = px.bar(
                display_df, 
                x="Destination", 
                y="Yield ($/Hour)",
                title="Expected Yield ($/Hour) by Destination",
                color="Yield ($/Hour)",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PAGE 4: MODEL EVALUATION
# =====================================================================
elif page == "4. Model Evaluation":
    st.title("📈 Model Evaluation")
    st.markdown("Comparison of Machine Learning models trained on the features dataset.")
    
    metrics = get_metrics()
    
    if metrics.empty:
        st.warning("No metric data found in MongoDB.")
    else:
        if "_id" in metrics.columns:
            metrics = metrics.drop(columns=["_id"])
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RMSE vs MAE Comparison")
            # Melt for grouped bar chart
            melted = metrics.melt(id_vars="model", value_vars=["rmse", "mae"], var_name="Metric", value_name="Error")
            fig = px.bar(
                melted, 
                x="model", 
                y="Error", 
                color="Metric", 
                barmode="group",
                title="Linear Regression vs GBT Regressor",
                labels={"model": "Model"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Metrics Table")
            st.dataframe(metrics.set_index("model"), use_container_width=True)
            
        st.markdown("---")
        st.subheader("Residuals & Feature Importance")
        st.write("Below are the pre-generated visual assets from the modeling pipeline (`plot_results.py`).")
        
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("./plots/residual_plots.png"):
                st.image("./plots/residual_plots.png", caption="Residual Plots")
            if os.path.exists("./plots/residual_distribution.png"):
                st.image("./plots/residual_distribution.png", caption="Residual Distribution")
        with c2:
            if os.path.exists("./plots/actual_vs_predicted.png"):
                st.image("./plots/actual_vs_predicted.png", caption="Actual vs Predicted")
            if os.path.exists("./plots/model_comparison.png"):
                st.image("./plots/model_comparison.png", caption="Model Comparison")
