from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

spark = SparkSession.builder.appName("NYC_Clean_Person1").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

try:
    print(">>> Loading taxi parquet...")
    taxi = spark.read.parquet("hdfs:///user/avn2049_nyu_edu/data/raw/2025/*/*.parquet")
    raw_count = taxi.count()
    print(f">>> Raw row count: {raw_count}")

    print(">>> Loading zone CSV...")
    zones = spark.read.option("header", "true").csv("hdfs:///user/avn2049_nyu_edu/data/taxi_zone_lookup.csv")
    print(">>> Cleaning...")
    cleaned = taxi.filter(
        col("tpep_pickup_datetime").isNotNull() &
        col("tpep_dropoff_datetime").isNotNull() &
        (col("fare_amount") > 0) &
        (col("fare_amount") < 500) &
        (col("trip_distance") > 0)
    )
    clean_count = cleaned.count()
    print(f">>> Cleaned row count: {clean_count}")
    print(f">>> Rows removed: {raw_count - clean_count}")

    print(">>> Joining with zone lookup...")
    final_output = cleaned.join(zones, cleaned.PULocationID == zones.LocationID, "left")
    
    output_path = "hdfs:///user/avn2049_nyu_edu/data/cleaned/2025_full_year"
    print(f">>> Writing to {output_path} ...")
    final_output.write.mode("overwrite").parquet(output_path)

    print("=========================================")
    print("SUCCESS: Data saved to " + output_path)
    print(f"FINAL ROW COUNT: {final_output.count()}")
    print("=========================================")

except Exception as e:
    import traceback
    print(">>> ERROR:")
    traceback.print_exc()

spark.stop()
