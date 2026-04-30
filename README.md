# NYC-Urban-Pulse: 2025 Taxi Data Analysis

## Project Overview
This project analyzes over 48 million records of NYC Yellow Taxi trip data from 2025 to uncover urban mobility patterns. We use a Big Data stack (Google Cloud Dataproc, HDFS, and Apache Spark) to process, clean, and analyze the data at scale.

## Team Roles
* **Anoop Navile (@anupvna):** Data Engineering & Ingestion (Phase 1)
* **Krishna Nikhil Kindi:** Machine Learning & Predictive Modeling (Phase 2)
* **Astha Vairat:** Data Visualization & Dashboarding (Phase 3)

---

## Phase 1: Data Engineering 
### Data Ingestion
- **Source:** NYC TLC Trip Record Data (Parquet format)
- **Scope:** Full year 2025 (January - December)
- **Storage:** Data was ingested from CloudFront and stored in a partitioned HDFS directory structure: `/user/avn2049_nyu_edu/data/raw/2025/{month}/`

### Data Cleaning & Transformation
The raw dataset contained nearly 49 million rows. A PySpark cleaning pipeline was developed to ensure data quality:
- **Filtering:** Removed records with null values, zero/negative trip distances, and unrealistic fare amounts.
- **Enrichment:** Performed a distributed join with the `taxi_zone_lookup.csv` to map `PULocationID` to actual Borough and Zone names (e.g., Manhattan, JFK Airport).
- **Final Output:** Cleaned data is saved in Parquet format on HDFS for Phase 2.

### Data Quality Report
| Metric | Value |
| :--- | :--- |
| **Raw 2025 Rows** | 48,722,602 |
| **Cleaned 2025 Rows** | 44,701,945 |
| **Outliers/Bad Rows Removed** | 4,020,657 |

---

## Phase 2: Analytics & Machine Learning

### Feature Engineering (`feature_engineering.py`)
Built zone-hour level aggregations from the cleaned 44.7M trip records using PySpark:
- **Aggregations:** Total trips, total revenue, average fare, average duration, average distance, and demand-to-supply ratio per zone per hour.
- **Time Features:** Hour of day, day of week, month, weekend flag, and US holiday flag.
- **Weather Integration:** Joined with NOAA hourly weather data (temperature, precipitation, wind speed) and categorized into weather buckets (clear, rainy, cold, hot).
- **Output:** Feature table saved to HDFS at `/user/avn2049_nyu_edu/data/features/zone_hour_features`

### Predictive Modeling (`modeling.py`)
Trained two fare prediction models using Spark MLlib on the feature table:
- **Linear Regression:** Baseline model with regularization tuning via 5-fold cross-validation.
- **Gradient Boosted Trees (GBTRegressor):** Advanced model with hyperparameter tuning (max depth, step size) via 5-fold cross-validation.
- **Evaluation:** 80/20 train/test split. Both models evaluated on RMSE and MAE.
- **Outputs:** Feature importance chart (GBT), residual plots for both models, and saved model artifacts.

### Top-K Destination Recommender (`recommender.py`)
Batch recommender system to answer: *"I'm a taxi driver in Zone X at hour H in weather Y — where should I go?"*
- Computes expected hourly revenue for every origin-destination-hour-weather combination.
- Filters low-sample routes (minimum 5 trips) for reliability.
- Ranks and stores the **Top 5 highest-yield destinations** per (origin_zone, hour, weather_bucket).
- Results written to **MongoDB** collections (`zone_aggregates`, `predictions`, `recommendations`) for Phase 3 dashboard.

### Visualization (`plot_results.py`)
Generates report-ready charts from model outputs:
- Residual plots for both Linear Regression and GBT models
- Feature importance bar chart from GBT
- RMSE vs MAE model comparison chart

---

## Technical Stack
- **Cloud Infrastructure:** Google Cloud Platform (GCP)
- **Cluster Management:** Dataproc
- **Distributed Storage:** HDFS
- **Processing Engine:** Apache Spark (PySpark)
- **ML Framework:** Spark MLlib (LinearRegression, GBTRegressor)
- **Database:** MongoDB (serving layer for dashboard)
- **Visualization:** Matplotlib
- **Version Control:** Git/GitHub
