from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PeekData").getOrCreate()
# Load your big cleaned dataset
df = spark.read.parquet("hdfs:///user/avn2049_nyu_edu/data/cleaned/2025_full_year")
# Show the first 5 rows, but vertical=True so you can read every column easily
df.show(5, vertical=True)
spark.stop()
