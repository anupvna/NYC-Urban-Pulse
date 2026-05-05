"""
mongo_loader.py — Person 3 (Serving Layer)
==========================================
Reads the locally-downloaded parquet outputs from Phase 2 and loads them
into MongoDB collections with compound indexes for fast dashboard queries.

Collections created:
  • zone_aggregates   — zone-hour features for the heatmap
  • predictions       — fare prediction samples (actual + predicted)
  • recommendations   — Top-K destination table from Person 2

Run:
    python3 mongo_loader.py
"""

import os
import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError
import time

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
MONGO_URI   = "mongodb://localhost:27017"
DB_NAME     = "nyc_urban_pulse"

FEATURES_DIR         = "./features_parquet"
RESIDUALS_DIR        = "./residuals_parquet"
METRICS_DIR          = "./metrics_parquet"
RECOMMENDATIONS_DIR  = "./recommendations_parquet"
PREDICTIONS_GRID_DIR = "./predictions_grid_parquet"

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def connect():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")          # fail fast if MongoDB is down
    print(f"✅ Connected to MongoDB at {MONGO_URI}")
    return client

def load_collection(db, collection_name, df, indexes):
    """Drop + recreate a collection, insert records, build indexes."""
    col = db[collection_name]
    col.drop()
    print(f"   ↳ Inserting {len(df):,} rows into '{collection_name}' …", end="", flush=True)
    t0 = time.time()
    records = df.to_dict("records")
    try:
        col.insert_many(records, ordered=False)
    except BulkWriteError as bwe:
        print(f"\n   ⚠  Bulk write partial error: {bwe.details['nInserted']} inserted")
    elapsed = time.time() - t0
    print(f" done in {elapsed:.1f}s")

    for idx_spec, unique in indexes:
        col.create_index(idx_spec, unique=unique)
        names = ", ".join(f[0] for f in idx_spec)
        print(f"   ↳ Index created on ({names})")

    count = col.count_documents({})
    print(f"   ↳ Collection '{collection_name}' has {count:,} documents")
    return col

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  NYC Urban Pulse — MongoDB Loader (Person 3)")
    print("=" * 60)

    client = connect()
    db = client[DB_NAME]

    # ── 1. zone_aggregates ─────────────────────────────────────────
    print("\n[1/4] Loading zone_aggregates …")
    feat = pd.read_parquet(FEATURES_DIR)
    feat = feat.rename(columns={"PULocationID": "zone_id"})
    # Aggregate per zone+hour across all dates for the heatmap baseline
    agg = (feat.groupby(["zone_id", "Zone", "Borough", "pickup_hour"], as_index=False)
               .agg(
                   total_trips    =("total_trips",    "sum"),
                   total_revenue  =("total_revenue",  "sum"),
                   avg_fare       =("avg_fare",        "mean"),
                   avg_duration   =("avg_duration_min","mean"),
                   avg_distance   =("avg_distance",   "mean"),
                   avg_tip        =("avg_tip",         "mean"),
                   is_weekend_pct =("is_weekend",     "mean"),
               )
              )
    agg = agg.rename(columns={"pickup_hour": "hour_bucket"})
    agg["avg_fare"]     = agg["avg_fare"].round(2)
    agg["avg_duration"] = agg["avg_duration"].round(2)
    agg["avg_distance"] = agg["avg_distance"].round(3)
    agg["avg_tip"]      = agg["avg_tip"].round(2)

    load_collection(
        db, "zone_aggregates", agg,
        indexes=[
            ([("zone_id", ASCENDING), ("hour_bucket", ASCENDING)], False),
            ([("Borough", ASCENDING)], False),
        ]
    )

    # ── 2. predictions ─────────────────────────────────────────────
    print("\n[2/4] Loading predictions …")
    res = pd.read_parquet(RESIDUALS_DIR)
    # Sample to keep MongoDB lean (keep up to 100k per model)
    res = (res.groupby("model", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), 100_000), random_state=42)))
    res = res.reset_index(drop=True)
    res["actual"]     = res["actual"].round(2)
    res["predicted"]  = res["predicted"].round(2)
    res["residual"]   = res["residual"].round(2)

    load_collection(
        db, "predictions", res,
        indexes=[
            ([("model", ASCENDING)], False),
        ]
    )

    # ── metrics sub-document ───────────────────────────────────────
    metrics_df = pd.read_parquet(METRICS_DIR)
    db["model_metrics"].drop()
    db["model_metrics"].insert_many(metrics_df.to_dict("records"))
    print("   ↳ Inserted model_metrics (2 documents)")

    # ── 3. recommendations ─────────────────────────────────────────
    print("\n[3/4] Loading recommendations …")
    reco = pd.read_parquet(RECOMMENDATIONS_DIR)
    reco = reco.rename(columns={"pickup_hour": "hour_bucket"})
    reco["avg_revenue"]      = reco["avg_revenue"].round(2)
    reco["avg_distance"]     = reco["avg_distance"].round(3)
    reco["revenue_per_hour"] = reco["revenue_per_hour"].round(2)

    load_collection(
        db, "recommendations", reco,
        indexes=[
            ([("origin_zone", ASCENDING), ("hour_bucket", ASCENDING)], False),
            ([("origin_zone", ASCENDING), ("hour_bucket", ASCENDING), ("rank", ASCENDING)], False),
        ]
    )

    # ── 4. predictions_grid ────────────────────────────────────────
    print("\n[4/4] Loading predictions_grid …")
    if os.path.isdir(PREDICTIONS_GRID_DIR):
        pred = pd.read_parquet(PREDICTIONS_GRID_DIR)
        pred["actual_fare"]    = pred["actual_fare"].round(2)
        pred["predicted_fare"] = pred["predicted_fare"].round(2)
        pred["pickup_date"]    = pred["pickup_date"].astype(str)

        load_collection(
            db, "predictions_grid", pred,
            indexes=[
                ([("zone_id", ASCENDING), ("pickup_date", ASCENDING), ("hour_bucket", ASCENDING)], False),
                ([("zone_id", ASCENDING), ("pickup_dow", ASCENDING), ("pickup_month", ASCENDING), ("hour_bucket", ASCENDING)], False),
            ]
        )
    else:
        print(f"   ⚠  {PREDICTIONS_GRID_DIR} not found — skipping predictions_grid.")
        print(f"      Run predict_grid.py on Dataproc and 'hdfs dfs -get' the output first.")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL COLLECTIONS LOADED")
    print("=" * 60)
    for name in ["zone_aggregates", "predictions", "model_metrics", "recommendations", "predictions_grid"]:
        n = db[name].count_documents({})
        print(f"  {name:<25}  {n:>10,} docs")

    # ── Quick query benchmark ──────────────────────────────────────
    print("\n[Benchmark] Running sample queries …")

    t0 = time.time()
    list(db["zone_aggregates"].find({"zone_id": 161, "hour_bucket": 9}))
    print(f"  zone_aggregates lookup (zone=161, hour=9): {(time.time()-t0)*1000:.1f} ms")

    t0 = time.time()
    list(db["recommendations"].find({"origin_zone": 161, "hour_bucket": 9}).sort("rank", 1).limit(5))
    print(f"  recommendations lookup (zone=161, hour=9): {(time.time()-t0)*1000:.1f} ms")

    if db["predictions_grid"].count_documents({}) > 0:
        t0 = time.time()
        db["predictions_grid"].find_one({"zone_id": 161, "pickup_date": "2025-06-15", "hour_bucket": 9})
        print(f"  predictions_grid lookup (zone=161, date=2025-06-15, hour=9): {(time.time()-t0)*1000:.1f} ms")

    client.close()
    print("\n✅ Done. MongoDB is ready for the dashboard.")

if __name__ == "__main__":
    main()
