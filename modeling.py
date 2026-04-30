from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, abs as spark_abs
import json

spark = SparkSession.builder \
    .appName("NYC_Modeling_Person2") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# =============================================================
# PATHS
# =============================================================
FEATURE_PATH = "hdfs:///user/avn2049_nyu_edu/data/features/zone_hour_features"
MODEL_OUTPUT  = "hdfs:///user/avn2049_nyu_edu/data/models/"
RESULTS_PATH  = "hdfs:///user/avn2049_nyu_edu/data/model_results/"

# =============================================================
# 1. LOAD FEATURE TABLE
# =============================================================
print(">>> Loading feature table...")
df = spark.read.parquet(FEATURE_PATH)
print(f">>> Loaded {df.count()} rows")

# =============================================================
# 2. PREPARE FEATURES FOR ML
# =============================================================
print(">>> Preparing features...")

# Index the Borough column (categorical -> numeric)
borough_indexer = StringIndexer(inputCol="Borough", outputCol="borough_idx", handleInvalid="keep")
zone_indexer = StringIndexer(inputCol="Zone", outputCol="zone_idx", handleInvalid="keep")
weather_indexer = StringIndexer(inputCol="weather_bucket", outputCol="weather_idx", handleInvalid="keep")

feature_columns = [
    "PULocationID", "pickup_hour", "pickup_dow", "pickup_month",
    "is_weekend", "is_holiday",
    "temperature", "precipitation", "wind_speed",
    "total_trips", "avg_distance", "avg_duration_min",
    "borough_idx", "zone_idx", "weather_idx"
]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")

# Target variable
TARGET = "avg_fare"

# Drop rows with null target
df = df.filter(col(TARGET).isNotNull())

# =============================================================
# 3. TRAIN/TEST SPLIT
# =============================================================
print(">>> Splitting data 80/20...")
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
print(f">>> Train: {train_data.count()}, Test: {test_data.count()}")

# =============================================================
# 4. MODEL 1: LINEAR REGRESSION
# =============================================================
print(">>> Training Linear Regression...")

lr = LinearRegression(featuresCol="features", labelCol=TARGET, maxIter=100)
lr_pipeline = Pipeline(stages=[borough_indexer, zone_indexer, weather_indexer, assembler, lr])

lr_paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

lr_evaluator = RegressionEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="rmse")

lr_cv = CrossValidator(
    estimator=lr_pipeline,
    estimatorParamMaps=lr_paramGrid,
    evaluator=lr_evaluator,
    numFolds=5,
    seed=42
)

lr_cv_model = lr_cv.fit(train_data)
lr_predictions = lr_cv_model.transform(test_data)

lr_rmse = lr_evaluator.evaluate(lr_predictions)
lr_evaluator_mae = RegressionEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="mae")
lr_mae = lr_evaluator_mae.evaluate(lr_predictions)

print(f">>> Linear Regression — RMSE: {lr_rmse:.4f}, MAE: {lr_mae:.4f}")

# =============================================================
# 5. MODEL 2: GRADIENT BOOSTED TREES
# =============================================================
print(">>> Training Gradient Boosted Trees...")

gbt = GBTRegressor(featuresCol="features", labelCol=TARGET, maxIter=100, seed=42)
gbt_pipeline = Pipeline(stages=[borough_indexer, zone_indexer, weather_indexer, assembler, gbt])

gbt_paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 8]) \
    .addGrid(gbt.stepSize, [0.05, 0.1]) \
    .build()

gbt_cv = CrossValidator(
    estimator=gbt_pipeline,
    estimatorParamMaps=gbt_paramGrid,
    evaluator=lr_evaluator,
    numFolds=5,
    seed=42
)

gbt_cv_model = gbt_cv.fit(train_data)
gbt_predictions = gbt_cv_model.transform(test_data)

gbt_rmse = lr_evaluator.evaluate(gbt_predictions)
gbt_mae = lr_evaluator_mae.evaluate(gbt_predictions)

print(f">>> GBT Regressor    — RMSE: {gbt_rmse:.4f}, MAE: {gbt_mae:.4f}")

# =============================================================
# 6. FEATURE IMPORTANCE (GBT)
# =============================================================
print(">>> Feature Importances (GBT):")
gbt_model = gbt_cv_model.bestModel.stages[-1]
importances = gbt_model.featureImportances.toArray()

feature_importance_list = sorted(
    zip(feature_columns, importances),
    key=lambda x: x[1],
    reverse=True
)
for feat, imp in feature_importance_list:
    print(f"    {feat:25s} : {imp:.4f}")

# =============================================================
# 7. SAVE RESIDUALS FOR PLOTTING
# =============================================================
print(">>> Computing residuals...")

lr_residuals = lr_predictions.select(
    col(TARGET).alias("actual"),
    col("prediction").alias("predicted"),
    (col(TARGET) - col("prediction")).alias("residual"),
    lit("LinearRegression").alias("model")
)

gbt_residuals = gbt_predictions.select(
    col(TARGET).alias("actual"),
    col("prediction").alias("predicted"),
    (col(TARGET) - col("prediction")).alias("residual"),
    lit("GBTRegressor").alias("model")
)

all_residuals = lr_residuals.union(gbt_residuals)
all_residuals.write.mode("overwrite").parquet(RESULTS_PATH + "residuals")

# =============================================================
# 8. SAVE RESULTS SUMMARY
# =============================================================
results = {
    "linear_regression": {"rmse": lr_rmse, "mae": lr_mae},
    "gbt_regressor": {"rmse": gbt_rmse, "mae": gbt_mae},
    "feature_importances": {f: float(i) for f, i in feature_importance_list}
}

results_df = spark.createDataFrame([
    ("LinearRegression", lr_rmse, lr_mae),
    ("GBTRegressor", gbt_rmse, gbt_mae)
], ["model", "rmse", "mae"])

results_df.write.mode("overwrite").parquet(RESULTS_PATH + "metrics")
print(">>> Results saved.")

# Save the best GBT model
gbt_cv_model.bestModel.write().overwrite().save(MODEL_OUTPUT + "gbt_best")
lr_cv_model.bestModel.write().overwrite().save(MODEL_OUTPUT + "lr_best")
print(">>> Models saved.")

# =============================================================
# 9. PRINT FINAL SUMMARY
# =============================================================
print("\n" + "=" * 50)
print("       MODEL COMPARISON RESULTS")
print("=" * 50)
print(f"  {'Model':<25} {'RMSE':>10} {'MAE':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10}")
print(f"  {'Linear Regression':<25} {lr_rmse:>10.4f} {lr_mae:>10.4f}")
print(f"  {'GBT Regressor':<25} {gbt_rmse:>10.4f} {gbt_mae:>10.4f}")
print("=" * 50)

spark.stop()
