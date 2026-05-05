import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from datetime import date
import json
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


@st.cache_resource
def load_geojson():
    """Load NYC Taxi Zones GeoJSON if present in project root.
    
    Normalises the property key to 'LocationID' (int) so it matches
    the zone_id column coming from MongoDB regardless of the original
    casing or type in the downloaded file.
    """
    path = "./taxi_zones.geojson"
    if os.path.exists(path):
        with open(path, "r") as f:
            geo = json.load(f)
        for feat in geo.get("features", []):
            props = feat.get("properties", {})
            lid = props.get("LocationID") or props.get("locationid") or props.get("LOCATIONID")
            if lid is not None:
                props["LocationID"] = int(lid)
        return geo
    return None


@st.cache_data
def get_gbt_residual_std():
    """Std deviation of GBT test-set residuals — used for prediction confidence band."""
    cursor = db.predictions.find({"model": "GBTRegressor"}, {"residual": 1, "_id": 0})
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return None
    return float(df["residual"].std())


# Spark dayofweek convention: Sun=1, Mon=2, ..., Sat=7
PY_DOW_TO_SPARK = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 1}
SPARK_DOW_NAMES = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
                   5: "Thursday", 6: "Friday", 7: "Saturday"}


@st.cache_data
def get_exact_prediction(zone_id, pickup_date_str, hour_bucket):
    """Look up a precomputed GBT prediction for a real 2025 (zone, date, hour)."""
    return db.predictions_grid.find_one({
        "zone_id": zone_id,
        "pickup_date": pickup_date_str,
        "hour_bucket": hour_bucket
    })


@st.cache_data
def get_forecast_prediction(zone_id, spark_dow, month, hour_bucket):
    """Forecast for a non-2025 date by averaging predictions for matching dow+month from 2025."""
    cursor = db.predictions_grid.find(
        {"zone_id": zone_id, "pickup_dow": spark_dow,
         "pickup_month": month, "hour_bucket": hour_bucket},
        {"predicted_fare": 1, "_id": 0}
    )
    preds = [d["predicted_fare"] for d in cursor]
    if not preds:
        return None, 0
    return sum(preds) / len(preds), len(preds)


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

    df_zones = get_zone_data(selected_hour)

    if df_zones.empty:
        st.info("No data available for this hour.")
    else:
        geojson = load_geojson()

        if geojson is not None:
            st.subheader(f"NYC Taxi Zones — {color_metric.replace('_', ' ').title()} at {selected_hour}:00")
            fig = px.choropleth_mapbox(
                df_zones,
                geojson=geojson,
                locations="zone_id",
                featureidkey="properties.LocationID",
                color=color_metric,
                color_continuous_scale="Viridis",
                mapbox_style="carto-positron",
                zoom=9,
                center={"lat": 40.7128, "lon": -74.0060},
                opacity=0.7,
                hover_name="Zone",
                hover_data={"Borough": True, color_metric: ":.2f", "zone_id": False},
            )
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 Place `taxi_zones.geojson` in the project root to enable the NYC choropleth map. Showing top-zones bar chart instead.")
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
    st.markdown("Predicted fares from the trained **Gradient Boosted Tree** model. "
                "Pick a 2025 date to compare prediction vs actual; pick a 2026 date for forecast mode.")

    zone_list = get_zone_list()

    if not zone_list:
        st.warning("No zone data available to make predictions.")
    elif db.predictions_grid.count_documents({}) == 0:
        st.warning("⚠️ `predictions_grid` collection is empty. Run `predict_grid.py` on the cluster, "
                   "then `hdfs dfs -get` the output and re-run `mongo_loader.py`.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            origin = st.selectbox(
                "Pickup Zone",
                options=[z[0] for z in zone_list],
                format_func=lambda x: next(z[1] for z in zone_list if z[0] == x)
            )
        with col2:
            pickup_date = st.date_input(
                "Pickup Date",
                value=date(2025, 6, 15),
                min_value=date(2025, 1, 1),
                max_value=date(2026, 12, 31)
            )
        with col3:
            hour = st.slider("Hour of Day", 0, 23, 12)

        st.markdown("---")

        residual_std = get_gbt_residual_std() or 0.0
        is_2025 = pickup_date.year == 2025

        if is_2025:
            doc = get_exact_prediction(origin, pickup_date.strftime("%Y-%m-%d"), hour)

            if doc:
                predicted = doc["predicted_fare"]
                actual = doc["actual_fare"]
                error = predicted - actual
                lower = predicted - 1.96 * residual_std
                upper = predicted + 1.96 * residual_std

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Predicted Fare (GBT)", f"${predicted:.2f}")
                    st.caption(f"95% CI: ${lower:.2f} – ${upper:.2f}")
                with c2:
                    st.metric("Actual Fare (Historical)", f"${actual:.2f}")
                with c3:
                    err_pct = (abs(error) / actual * 100) if actual > 0 else 0
                    st.metric("Prediction Error", f"${error:+.2f}", f"{err_pct:.1f}% off")

                day_name = SPARK_DOW_NAMES.get(doc.get("pickup_dow"), "?")
                weekend_str = "Weekend" if doc.get("is_weekend") else "Weekday"
                holiday_str = "🎉 US Holiday" if doc.get("is_holiday") else "Regular day"
                st.caption(f"📅 {day_name}, {pickup_date.strftime('%B %d, %Y')} • {weekend_str} • {holiday_str}")
            else:
                st.info("No data for this combination of zone, date, and hour. "
                        "Some (zone, hour) pairs had no trips on this specific date.")

        else:
            # 2026 — forecast mode
            spark_dow = PY_DOW_TO_SPARK[pickup_date.weekday()]
            predicted, n_samples = get_forecast_prediction(origin, spark_dow, pickup_date.month, hour)

            if predicted is not None:
                lower = predicted - 1.96 * residual_std
                upper = predicted + 1.96 * residual_std

                st.warning("⚠️ Forecast mode: 2026 has no actual data — projection based on 2025 patterns "
                           "for the same weekday and month.")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Forecast Fare", f"${predicted:.2f}")
                    st.caption(f"95% CI: ${lower:.2f} – ${upper:.2f}")
                with c2:
                    st.metric("Based on", f"{n_samples} matching 2025 days")

                day_name = SPARK_DOW_NAMES.get(spark_dow, "?")
                st.caption(f"📅 {day_name}, {pickup_date.strftime('%B %d, %Y')} • "
                           f"Same weekday + month as in training data")
            else:
                st.info("Not enough 2025 data with the same weekday + month to forecast this combination.")
                

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
