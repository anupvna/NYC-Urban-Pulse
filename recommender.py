from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, sum as spark_sum, row_number, when, lit
)
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("NYC_Recommender_Person2") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# =============================================================
# PATHS
# =============================================================
CLEANED_DATA_PATH = "hdfs:///user/kk6064_nyu_edu/data/taxi_copy"
FEATURE_PATH      = "hdfs:///user/kk6064_nyu_edu/data/features/zone_hour_features"
RESULTS_PATH      = "hdfs:///user/kk6064_nyu_edu/data/model_results/"

# =============================================================
# 1. LOAD CLEANED TRIP DATA (need origin->destination pairs)
# =============================================================
print(">>> Loading cleaned taxi data for OD pairs...")
taxi = spark.read.parquet(CLEANED_DATA_PATH)

from pyspark.sql.functions import (
    hour as spark_hour, unix_timestamp
)

taxi = taxi.withColumn("pickup_hour", spark_hour(col("tpep_pickup_datetime")))

# =============================================================
# 2. COMPUTE EXPECTED YIELD PER ORIGIN-DESTINATION-HOUR
# =============================================================
print(">>> Computing origin-destination yield matrix...")

od_yield = taxi.groupBy(
    col("PULocationID").alias("origin_zone"),
    col("DOLocationID").alias("dest_zone"),
    "pickup_hour"
).agg(
    avg("total_amount").alias("avg_revenue"),
    count("*").alias("trip_count"),
    avg("trip_distance").alias("avg_distance"),
    avg(
        col("total_amount") /
        (
            (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 3600.0
        )
    ).alias("revenue_per_hour")
)

# Filter out low-sample routes (need at least 5 trips to be a reliable recommendation)
od_yield = od_yield.filter(col("trip_count") >= 5)

# =============================================================
# 3. RANK TOP-K DESTINATIONS PER ORIGIN+HOUR
# =============================================================
print(">>> Ranking top 5 destinations per origin-hour combo...")

window = Window.partitionBy("origin_zone", "pickup_hour") \
    .orderBy(col("revenue_per_hour").desc())

top_k = od_yield.withColumn("rank", row_number().over(window)) \
    .filter(col("rank") <= 5)

top_k_count = top_k.count()
print(f">>> Total recommendations: {top_k_count}")
print(">>> Sample recommendations:")
top_k.show(20, truncate=False)

# =============================================================
# 4. SAVE TO HDFS (intermediate, before MongoDB)
# =============================================================
reco_hdfs_path = "hdfs:///user/kk6064_nyu_edu/data/recommendations/top_k"
top_k.write.mode("overwrite").parquet(reco_hdfs_path)
print(f">>> Saved recommendations to {reco_hdfs_path}")

# =============================================================
# 5. MONGODB WRITES — handled by Person 3 (dashboard)
# =============================================================
# Person 3 will read the HDFS parquet outputs and load into MongoDB
# for the dashboard. Data locations:
#   - Recommendations: hdfs:///user/kk6064_nyu_edu/data/recommendations/top_k
#   - Features:        hdfs:///user/kk6064_nyu_edu/data/features/zone_hour_features
#   - Model results:   hdfs:///user/kk6064_nyu_edu/data/model_results/

print("\n" + "=" * 50)
print("  RECOMMENDER PIPELINE COMPLETE")
print("=" * 50)

spark.stop()
