from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("QualityReport").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Pointing to your BIG 2025 dataset
raw = spark.read.parquet("hdfs:///user/avn2049_nyu_edu/data/raw/2025/*/*.parquet")
cleaned = spark.read.parquet("hdfs:///user/avn2049_nyu_edu/data/cleaned/2025_full_year")

print("\n" + "="*30)
print("   FINAL DATA QUALITY REPORT")
print("="*30)
print(f"Raw 2025 Rows:     {raw.count()}")
print(f"Cleaned 2025 Rows: {cleaned.count()}")
print(f"Bad Rows Removed:  {raw.count() - cleaned.count()}")
print("="*30 + "\n")

print("--- Data Schema  ---")
cleaned.printSchema()
spark.stop()
