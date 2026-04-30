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
CLEANED_DATA_PATH = "hdfs:///user/avn2049_nyu_edu/data/cleaned/2025_full_year"
WEATHER_DATA_PATH = "hdfs:///user/avn2049_nyu_edu/data/weather/"
OUTPUT_PATH       = "hdfs:///user/avn2049_nyu_edu/data/features/zone_hour_features"

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

# Round pickup time to the nearest hour for weather join
taxi = taxi.withColumn(
    "pickup_hour_rounded",
    from_unixtime(
        (unix_timestamp(col("tpep_pickup_datetime")) / 3600).cast("long") * 3600
    ).cast("timestamp")
)

# Trip duration in minutes
taxi = taxi.withColumn(
    "trip_duration_min",
    (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60.0
).filter(col("trip_duration_min") > 0)

# =============================================================
# 3. LOAD AND JOIN WEATHER DATA
# =============================================================
print(">>> Loading NOAA weather data...")
weather = spark.read.parquet(WEATHER_DATA_PATH)
print(">>> Weather schema:")
weather.printSchema()

# Rename weather columns to avoid ambiguity after join
# Adjust these column names based on actual NOAA schema
weather = weather.withColumn(
    "weather_hour",
    from_unixtime(
        (unix_timestamp(col("DATE")) / 3600).cast("long") * 3600
    ).cast("timestamp")
).select(
    col("weather_hour"),
    col("HourlyDryBulbTemperature").alias("temperature").cast(DoubleType()),
    col("HourlyPrecipitation").alias("precipitation").cast(DoubleType()),
    col("HourlyWindSpeed").alias("wind_speed").cast(DoubleType())
).dropDuplicates(["weather_hour"])

# Fill nulls in weather with 0 (no rain/wind) or reasonable defaults
weather = weather.fillna({"temperature": 55.0, "precipitation": 0.0, "wind_speed": 5.0})

print(">>> Joining taxi data with weather...")
taxi = taxi.join(weather, taxi.pickup_hour_rounded == weather.weather_hour, "left")

# Fill any remaining nulls from unmatched weather rows
taxi = taxi.fillna({"temperature": 55.0, "precipitation": 0.0, "wind_speed": 5.0})

# =============================================================
# 4. ZONE-HOUR AGGREGATIONS
# =============================================================
print(">>> Building zone-hour aggregations...")

zone_hour_features = taxi.groupBy(
    "PULocationID", "Borough", "Zone",
    "pickup_date", "pickup_hour", "pickup_dow", "pickup_month",
    "is_weekend", "is_holiday",
    "temperature", "precipitation", "wind_speed"
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
    col("total_trips") / lit(1)  # per zone-hour, this is the raw demand count
)

# Weather buckets for recommender (Person 3 needs these)
zone_hour_features = zone_hour_features.withColumn(
    "weather_bucket",
    when(col("precipitation") > 0.1, "rainy")
    .when(col("temperature") < 32, "cold")
    .when(col("temperature") > 85, "hot")
    .otherwise("clear")
)

# =============================================================
# 5. SAVE FEATURE TABLE
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
