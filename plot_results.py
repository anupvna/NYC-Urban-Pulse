"""
Run this AFTER modeling.py to generate charts for the final report.
Can run locally (not on Spark) — just needs the residuals CSV exported from HDFS.

Usage on cluster:
  1. Export residuals: hdfs dfs -get /user/kk6064_nyu_edu/data/model_results/residuals ./residuals_parquet
  2. Run: python plot_results.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

RESIDUALS_DIR = "./residuals_parquet"
OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(">>> Loading residuals...")
df = pd.read_parquet(RESIDUALS_DIR)

lr_df = df[df["model"] == "LinearRegression"]
gbt_df = df[df["model"] == "GBTRegressor"]

# =============================================================
# 1. RESIDUAL PLOTS
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(lr_df["predicted"], lr_df["residual"], alpha=0.3, s=5, c="steelblue")
axes[0].axhline(y=0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted Fare")
axes[0].set_ylabel("Residual")
axes[0].set_title("Linear Regression — Residuals")

axes[1].scatter(gbt_df["predicted"], gbt_df["residual"], alpha=0.3, s=5, c="darkorange")
axes[1].axhline(y=0, color="red", linestyle="--")
axes[1].set_xlabel("Predicted Fare")
axes[1].set_ylabel("Residual")
axes[1].set_title("GBT Regressor — Residuals")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/residual_plots.png", dpi=150)
print(f">>> Saved {OUTPUT_DIR}/residual_plots.png")

# =============================================================
# 2. FEATURE IMPORTANCE (manually paste from modeling.py output)
# Update these values after running modeling.py
# =============================================================
feature_importances = {
    "pickup_hour": 0.0,
    "pickup_dow": 0.0,
    "pickup_month": 0.0,
    "is_weekend": 0.0,
    "is_holiday": 0.0,
    "temperature": 0.0,
    "precipitation": 0.0,
    "wind_speed": 0.0,
    "total_trips": 0.0,
    "avg_distance": 0.0,
    "avg_duration_min": 0.0,
    "PULocationID": 0.0,
    "borough_idx": 0.0,
    "zone_idx": 0.0,
    "weather_idx": 0.0,
}

# Sort by importance
sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
names = [f[0] for f in sorted_features]
values = [f[1] for f in sorted_features]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(names[::-1], values[::-1], color="teal")
ax.set_xlabel("Importance")
ax.set_title("GBT Feature Importances")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
print(f">>> Saved {OUTPUT_DIR}/feature_importance.png")

# =============================================================
# 3. MODEL COMPARISON BAR CHART
# =============================================================
# Update these after running modeling.py
lr_rmse, lr_mae = 0.0, 0.0
gbt_rmse, gbt_mae = 0.0, 0.0

fig, ax = plt.subplots(figsize=(8, 5))
x = [0, 1]
width = 0.35
ax.bar([i - width/2 for i in x], [lr_rmse, gbt_rmse], width, label="RMSE", color="steelblue")
ax.bar([i + width/2 for i in x], [lr_mae, gbt_mae], width, label="MAE", color="darkorange")
ax.set_xticks(x)
ax.set_xticklabels(["Linear Regression", "GBT Regressor"])
ax.set_ylabel("Error")
ax.set_title("Model Comparison: RMSE vs MAE")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150)
print(f">>> Saved {OUTPUT_DIR}/model_comparison.png")

print("\n>>> All plots saved to ./plots/")
