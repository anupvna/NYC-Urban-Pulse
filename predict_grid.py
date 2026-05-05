"""
predict_grid.py — Person 2 (Inference)
=======================================
Loads the trained GBT pipeline model and runs inference on every row in the
zone-hour feature parquet. Outputs per-(zone, date, hour) predictions to HDFS
so the dashboard can serve real model predictions via MongoDB lookup.

No retraining, just a forward pass on the saved PipelineModel.

Run on Dataproc:
    spark-submit predict_grid.py

Then download to local before running mongo_loader.py:
    hdfs dfs -get /user/kk6064_nyu_edu/data/predictions/zone_hour_grid \\
        ./predictions_grid_parquet
"""

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("NYC_PredictGrid_Person2") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# =============================================================
# PATHS
# =============================================================
FEATURE_PATH = "hdfs:///user/kk6064_nyu_edu/data/features/zone_hour_features"
MODEL_PATH   = "hdfs:///user/kk6064_nyu_edu/data/models/gbt_best"
OUTPUT_PATH  = "hdfs:///user/kk6064_nyu_edu/data/predictions/zone_hour_grid"

# =============================================================
# 1. LOAD SAVED MODEL
# =============================================================
print(">>> Loading saved GBT pipeline model...")
model = PipelineModel.load(MODEL_PATH)

# =============================================================
# 2. LOAD FEATURE TABLE
# =============================================================
print(">>> Loading feature table...")
df = spark.read.parquet(FEATURE_PATH)
print(f">>> Feature rows: {df.count()}")

# =============================================================
# 3. RUN INFERENCE
# =============================================================
print(">>> Running model inference...")
predictions = model.transform(df)

result = predictions.select(
    col("PULocationID").alias("zone_id"),
    "Zone", "Borough",
    "pickup_date",
    col("pickup_hour").alias("hour_bucket"),
    "pickup_dow", "pickup_month",
    "is_weekend", "is_holiday",
    col("avg_fare").alias("actual_fare"),
    col("prediction").alias("predicted_fare")
)

print(">>> Sample predictions:")
result.show(10, truncate=False)

# =============================================================
# 4. SAVE
# =============================================================
print(f">>> Writing predictions to {OUTPUT_PATH}")
result.write.mode("overwrite").parquet(OUTPUT_PATH)

print("=" * 50)
print("  PREDICTION GRID COMPLETE")
print("=" * 50)

spark.stop()
