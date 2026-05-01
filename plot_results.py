"""
Run this AFTER modeling.py to generate charts for the final report.
Can run locally (not on Spark) — just needs parquet files exported from HDFS.

Usage on cluster:
  1. hdfs dfs -get /user/kk6064_nyu_edu/data/model_results/residuals ./residuals_parquet
  2. hdfs dfs -get /user/kk6064_nyu_edu/data/model_results/metrics ./metrics_parquet
  3. python plot_results.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

RESIDUALS_DIR = "./residuals_parquet"
METRICS_DIR = "./metrics_parquet"
OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# 1. RESIDUAL PLOTS
# =============================================================
print(">>> Loading residuals...")
df = pd.read_parquet(RESIDUALS_DIR)

lr_df = df[df["model"] == "LinearRegression"]
gbt_df = df[df["model"] == "GBTRegressor"]

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
# 2. MODEL COMPARISON BAR CHART
# =============================================================
print(">>> Loading metrics...")
metrics = pd.read_parquet(METRICS_DIR)

lr_row = metrics[metrics["model"] == "LinearRegression"].iloc[0]
gbt_row = metrics[metrics["model"] == "GBTRegressor"].iloc[0]

fig, ax = plt.subplots(figsize=(8, 5))
x = [0, 1]
width = 0.35
ax.bar([i - width/2 for i in x], [lr_row["rmse"], gbt_row["rmse"]], width, label="RMSE", color="steelblue")
ax.bar([i + width/2 for i in x], [lr_row["mae"], gbt_row["mae"]], width, label="MAE", color="darkorange")
ax.set_xticks(x)
ax.set_xticklabels(["Linear Regression", "GBT Regressor"])
ax.set_ylabel("Error")
ax.set_title("Model Comparison: RMSE vs MAE")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150)
print(f">>> Saved {OUTPUT_DIR}/model_comparison.png")

# =============================================================
# 3. ACTUAL vs PREDICTED SCATTER
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(lr_df["actual"], lr_df["predicted"], alpha=0.3, s=5, c="steelblue")
axes[0].plot([lr_df["actual"].min(), lr_df["actual"].max()],
             [lr_df["actual"].min(), lr_df["actual"].max()], "r--")
axes[0].set_xlabel("Actual Fare")
axes[0].set_ylabel("Predicted Fare")
axes[0].set_title("Linear Regression — Actual vs Predicted")

axes[1].scatter(gbt_df["actual"], gbt_df["predicted"], alpha=0.3, s=5, c="darkorange")
axes[1].plot([gbt_df["actual"].min(), gbt_df["actual"].max()],
             [gbt_df["actual"].min(), gbt_df["actual"].max()], "r--")
axes[1].set_xlabel("Actual Fare")
axes[1].set_ylabel("Predicted Fare")
axes[1].set_title("GBT Regressor — Actual vs Predicted")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted.png", dpi=150)
print(f">>> Saved {OUTPUT_DIR}/actual_vs_predicted.png")

# =============================================================
# 4. RESIDUAL DISTRIBUTION
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(lr_df["residual"], bins=50, color="steelblue", alpha=0.7, edgecolor="black")
axes[0].set_xlabel("Residual")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Linear Regression — Residual Distribution")

axes[1].hist(gbt_df["residual"], bins=50, color="darkorange", alpha=0.7, edgecolor="black")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].set_title("GBT Regressor — Residual Distribution")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/residual_distribution.png", dpi=150)
print(f">>> Saved {OUTPUT_DIR}/residual_distribution.png")

print("\n>>> All plots saved to ./plots/")
