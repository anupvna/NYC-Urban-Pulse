from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, hour, dayofweek, month, year, date_format,
    avg, sum as spark_sum, count, lit, when, round as spark_round,
    unix_timestamp, from_unixtime, to_timestamp
)
from pyspark.sql.types import IntegerType, DoubleType
import datetime

spark = SparkSession.builder \
    .appName("NYC_FeatureEngineering_Person2") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# =============================================================
# PATHS — adjust these if Anoop used different locations
# =============================================================
CLEANED_DATA_PATH = "hdfs:///user/kk6064_nyu_edu/data/taxi_copy"
OUTPUT_PATH       = "hdfs:///user/kk6064_nyu_edu/data/features/zone_hour_features"

# =============================================================
# 1. LOAD CLEANED TAXI DATA
# =============================================================
print(">>> Loading cleaned taxi data...")
taxi = spark.read.parquet(CLEANED_DATA_PATH)
print(f">>> Loaded {taxi.count()} rows")
taxi.printSchema()

# =============================================================
# 2. ADD TIME FEATURES
# =============================================================
print(">>> Adding time features...")

US_HOLIDAYS_2025 = [
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13",
    "2025-11-11", "2025-11-27", "2025-12-25"
]

taxi = taxi.withColumn("pickup_hour", hour(col("tpep_pickup_datetime"))) \
    .withColumn("pickup_dow", dayofweek(col("tpep_pickup_datetime"))) \
    .withColumn("pickup_month", month(col("tpep_pickup_datetime"))) \
    .withColumn("pickup_date", date_format(col("tpep_pickup_datetime"), "yyyy-MM-dd")) \
    .withColumn("is_weekend", when(dayofweek(col("tpep_pickup_datetime")).isin(1, 7), 1).otherwise(0)) \
    .withColumn("is_holiday", when(date_format(col("tpep_pickup_datetime"), "yyyy-MM-dd").isin(US_HOLIDAYS_2025), 1).otherwise(0))

# Trip duration in minutes
taxi = taxi.withColumn(
    "trip_duration_min",
    (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60.0
).filter(col("trip_duration_min") > 0)

# =============================================================
# 3. ZONE-HOUR AGGREGATIONS
# =============================================================
print(">>> Building zone-hour aggregations...")

zone_hour_features = taxi.groupBy(
    "PULocationID", "Borough", "Zone",
    "pickup_date", "pickup_hour", "pickup_dow", "pickup_month",
    "is_weekend", "is_holiday"
).agg(
    count("*").alias("total_trips"),
    spark_sum("total_amount").alias("total_revenue"),
    avg("fare_amount").alias("avg_fare"),
    avg("trip_duration_min").alias("avg_duration_min"),
    avg("trip_distance").alias("avg_distance"),
    avg("tip_amount").alias("avg_tip"),
    spark_sum("passenger_count").alias("total_passengers")
)

# Demand-to-supply ratio (trips per unique hour — higher means busier zone)
zone_hour_features = zone_hour_features.withColumn(
    "demand_supply_ratio",
    col("total_trips") / lit(1)
)

# =============================================================
# 4. SAVE FEATURE TABLE
# =============================================================
print(f">>> Feature table row count: {zone_hour_features.count()}")
print(">>> Sample rows:")
zone_hour_features.show(10, truncate=False)

print(f">>> Saving feature table to {OUTPUT_PATH} ...")
zone_hour_features.write.mode("overwrite").parquet(OUTPUT_PATH)

print("=========================================")
print("SUCCESS: Feature table saved!")
print("=========================================")

spark.stop()
