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

## Technical Stack
- **Cloud Infrastructure:** Google Cloud Platform (GCP)
- **Cluster Management:** Dataproc
- **Distributed Storage:** HDFS
- **Processing Engine:** Apache Spark (PySpark)
- **Version Control:** Git/GitHub
