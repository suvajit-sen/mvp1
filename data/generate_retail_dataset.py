"""
Retail Sample Dataset Generator for Business Data Quality POC
=============================================================
Generates synthetic retail data across multiple tables with
intentional business data quality problems embedded for testing.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)
STORES = ["S001", "S002", "S003", "S004", "S005"]
CATEGORIES = ["Electronics", "Apparel", "Grocery", "Home & Kitchen", "Sports"]
CHANNELS = ["In-Store", "Online", "Mobile App"]

PRODUCTS = {
    "P001": {"name": "Laptop",         "category": "Electronics",    "base_price": 850, "base_cost": 600},
    "P002": {"name": "T-Shirt",        "category": "Apparel",        "base_price": 25,  "base_cost": 8},
    "P003": {"name": "Rice (5kg)",     "category": "Grocery",        "base_price": 12,  "base_cost": 7},
    "P004": {"name": "Coffee Maker",   "category": "Home & Kitchen", "base_price": 120, "base_cost": 70},
    "P005": {"name": "Running Shoes",  "category": "Sports",         "base_price": 95,  "base_cost": 45},
    "P006": {"name": "Smartphone",     "category": "Electronics",    "base_price": 699, "base_cost": 450},
    "P007": {"name": "Jeans",          "category": "Apparel",        "base_price": 60,  "base_cost": 22},
    "P008": {"name": "Cooking Oil (2L)", "category": "Grocery",      "base_price": 8,   "base_cost": 4},
    "P009": {"name": "Blender",        "category": "Home & Kitchen", "base_price": 75,  "base_cost": 38},
    "P010": {"name": "Yoga Mat",       "category": "Sports",         "base_price": 40,  "base_cost": 18},
}

SEASONAL_FACTORS = {
    1: 0.80, 2: 0.75, 3: 0.85, 4: 0.90, 5: 0.95, 6: 1.00,
    7: 1.05, 8: 1.10, 9: 1.00, 10: 1.05, 11: 1.30, 12: 1.60,
}

CATEGORY_SEASONAL = {
    "Electronics":    {11: 1.8, 12: 2.2, 1: 0.6},
    "Apparel":        {6: 1.4,  7: 1.5, 12: 1.6, 1: 0.7},
    "Grocery":        {11: 1.2, 12: 1.3},
    "Home & Kitchen": {11: 1.5, 12: 1.8},
    "Sports":         {3: 1.3,  4: 1.4, 5: 1.5},
}


def _date_range():
    dates = []
    cur = START_DATE
    while cur <= END_DATE:
        dates.append(cur)
        cur += timedelta(days=1)
    return dates


# ── Table 1: Products master ────────────────────────────────────────────────
def generate_products() -> pd.DataFrame:
    rows = []
    for pid, info in PRODUCTS.items():
        rows.append({
            "product_id":   pid,
            "product_name": info["name"],
            "category":     info["category"],
            "base_price":   info["base_price"],
            "base_cost":    info["base_cost"],
            "supplier_id":  f"SUP{np.random.randint(1, 11):03d}",
        })
    return pd.DataFrame(rows)


# ── Table 2: Daily Orders received ─────────────────────────────────────────
def generate_orders() -> pd.DataFrame:
    rows = []
    for date in _date_range():
        month = date.month
        is_weekend = date.weekday() >= 5
        for store in STORES:
            for pid, info in PRODUCTS.items():
                cat = info["category"]
                season = CATEGORY_SEASONAL.get(cat, {}).get(month, SEASONAL_FACTORS[month])
                base_orders = np.random.randint(20, 80)
                orders = max(0, int(base_orders * season * (1.25 if is_weekend else 1.0)
                                    + np.random.normal(0, 5)))
                rows.append({
                    "date":       date.strftime("%Y-%m-%d"),
                    "store_id":   store,
                    "product_id": pid,
                    "orders_received": orders,
                })

    df = pd.DataFrame(rows)

    # ── Inject DQ Issue #1: Orders missing on random days (data completeness) ──
    missing_idx = df.sample(frac=0.015, random_state=1).index
    df.loc[missing_idx, "orders_received"] = np.nan

    return df


# ── Table 3: Daily Sales transactions ──────────────────────────────────────
def generate_sales(orders_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, order_row in orders_df.iterrows():
        orders = order_row["orders_received"]
        if pd.isna(orders):
            continue
        orders = int(orders)
        pid = order_row["product_id"]
        info = PRODUCTS[pid]
        date = order_row["date"]
        store = order_row["store_id"]

        # fulfillment rate (normally ~90 %)
        fulfillment = np.random.uniform(0.82, 0.98)
        sales_qty = max(0, int(orders * fulfillment + np.random.normal(0, 3)))

        base_price = info["base_price"]
        # small price variance per store
        price = round(base_price * np.random.uniform(0.97, 1.03), 2)
        discount_pct = round(np.random.uniform(0, 0.15), 3)
        net_price = round(price * (1 - discount_pct), 2)
        sales_amount = round(sales_qty * net_price, 2)
        returns_qty = max(0, int(sales_qty * np.random.uniform(0, 0.05)))

        rows.append({
            "date":           date,
            "store_id":       store,
            "product_id":     pid,
            "channel":        np.random.choice(CHANNELS, p=[0.5, 0.35, 0.15]),
            "sales_qty":      sales_qty,
            "unit_price":     price,
            "discount_pct":   discount_pct,
            "sales_amount":   sales_amount,
            "returns_qty":    returns_qty,
            "cost_per_unit":  info["base_cost"],
        })

    df = pd.DataFrame(rows)
    df["gross_margin"] = (
        (df["unit_price"] * (1 - df["discount_pct"]) - df["cost_per_unit"]) * df["sales_qty"]
    ).round(2)

    # ────────────────────────────────────────────────────────────────────────
    # Inject DQ Issues into sales
    # ────────────────────────────────────────────────────────────────────────

    # Issue #2: High sales NOT backed by high orders → contextual anomaly
    anomaly_idx = df[
        (df["product_id"] == "P001") &
        (df["store_id"] == "S002") &
        (df["date"].between("2023-06-01", "2023-06-10"))
    ].index
    df.loc[anomaly_idx, "sales_qty"] = (df.loc[anomaly_idx, "sales_qty"] * 4).astype(int)
    df.loc[anomaly_idx, "sales_amount"] = (
        df.loc[anomaly_idx, "sales_qty"] * df.loc[anomaly_idx, "unit_price"]
        * (1 - df.loc[anomaly_idx, "discount_pct"])
    ).round(2)

    # Issue #3: Returns spike with no corresponding sales spike
    returns_spike_idx = df[
        (df["product_id"] == "P002") &
        (df["store_id"] == "S003") &
        (df["date"].between("2023-09-01", "2023-09-07"))
    ].index
    df.loc[returns_spike_idx, "returns_qty"] = (
        df.loc[returns_spike_idx, "sales_qty"] * 0.60
    ).astype(int)

    # Issue #4: Deep discount with no revenue uplift – 2-month window for statistical power
    discount_anomaly_idx = df[
        (df["product_id"] == "P006") &
        (df["store_id"].isin(["S001", "S004"])) &
        (df["date"].between("2023-09-01", "2023-10-31"))
    ].index
    df.loc[discount_anomaly_idx, "discount_pct"] = 0.40
    # sales_qty stays the same → revenue drops without volume gain

    # Issue #5: Price inconsistency – same product, wildly different price in one store
    price_anomaly_idx = df[
        (df["product_id"] == "P005") &
        (df["store_id"] == "S005") &
        (df["date"] > "2023-07-01")
    ].index
    df.loc[price_anomaly_idx, "unit_price"] = df.loc[price_anomaly_idx, "unit_price"] * 1.85

    # Issue #6: Negative gross margin rows (selling below cost)
    df["cost_per_unit"] = df["cost_per_unit"].astype(float)
    neg_margin_idx = df[
        (df["product_id"] == "P008") &
        (df["date"].between("2023-05-01", "2023-05-15"))
    ].index
    df.loc[neg_margin_idx, "cost_per_unit"] = (df.loc[neg_margin_idx, "unit_price"] * 1.40).round(2)

    return df


# ── Table 4: Daily Inventory ────────────────────────────────────────────────
def generate_inventory(sales_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for store in STORES:
        for pid in PRODUCTS:
            opening_stock = np.random.randint(200, 500)
            for date in _date_range():
                date_str = date.strftime("%Y-%m-%d")
                sold = sales_df[
                    (sales_df["store_id"] == store) &
                    (sales_df["product_id"] == pid) &
                    (sales_df["date"] == date_str)
                ]["sales_qty"].sum()

                restocked = 0
                # restock every 3 days to keep inventory healthy
                if date.weekday() in (0, 2, 4):
                    restocked = np.random.randint(80, 200)

                closing_stock = max(0, opening_stock - int(sold) + restocked)
                rows.append({
                    "date":          date_str,
                    "store_id":      store,
                    "product_id":    pid,
                    "opening_stock": opening_stock,
                    "units_sold":    int(sold),
                    "restocked_qty": restocked,
                    "closing_stock": closing_stock,
                })
                opening_stock = closing_stock

    df = pd.DataFrame(rows)

    # Issue #7: Stock-out despite normal orders (inventory accuracy problem)
    stockout_idx = df[
        (df["product_id"] == "P004") &
        (df["store_id"] == "S001") &
        (df["date"].between("2023-08-01", "2023-08-14"))
    ].index
    df.loc[stockout_idx, "closing_stock"] = 0

    # Issue #8: Ghost inventory (closing stock doesn't reconcile with opening - sold + restocked)
    ghost_idx = df[
        (df["product_id"] == "P009") &
        (df["store_id"] == "S002") &
        (df["date"].between("2023-11-01", "2023-11-07"))
    ].index
    df.loc[ghost_idx, "closing_stock"] = df.loc[ghost_idx, "opening_stock"] + 999

    return df


# ── Table 5: Customer Transactions ─────────────────────────────────────────
def generate_customers(n_customers: int = 500) -> pd.DataFrame:
    customer_ids = [f"C{i:05d}" for i in range(1, n_customers + 1)]
    segments = np.random.choice(["Premium", "Regular", "Occasional"], n_customers,
                                p=[0.15, 0.55, 0.30])
    clv_base = {"Premium": 1500, "Regular": 400, "Occasional": 80}
    rows = []
    for cid, seg in zip(customer_ids, segments):
        base = clv_base[seg]
        rows.append({
            "customer_id":       cid,
            "segment":           seg,
            "acquisition_month": np.random.randint(1, 13),
            "clv_estimate":      round(base * np.random.uniform(0.7, 1.4), 2),
            "avg_order_value":   round(base / 12 * np.random.uniform(0.8, 1.3), 2),
            "churn_risk_score":  round(np.random.uniform(0, 1), 3),
            "preferred_channel": np.random.choice(CHANNELS),
            "store_id":          np.random.choice(STORES),
        })
    df = pd.DataFrame(rows)

    # Issue #9: CLV anomaly – high CLV but very low avg order value (inconsistency)
    # Inject into a small minority (< 10%) so they are genuine statistical outliers
    anomaly_cust = df[df["segment"] == "Premium"].sample(5, random_state=7).index
    df.loc[anomaly_cust, "avg_order_value"] = round(np.random.uniform(1.5, 3.5), 2)

    # Issue #10: Churn risk above 0.9 for newly acquired (month >= 10) Premium customers
    new_premium = df[(df["segment"] == "Premium") & (df["acquisition_month"] >= 10)].index
    df.loc[new_premium, "churn_risk_score"] = round(np.random.uniform(0.91, 0.99), 3)

    return df


# ── Table 6: Supply-chain fulfillment ──────────────────────────────────────
def generate_supply_chain(orders_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        orders_df.groupby(["date", "store_id"])["orders_received"]
        .sum().reset_index()
        .rename(columns={"orders_received": "total_orders"})
    )
    agg["expected_lead_days"] = 2
    agg["actual_lead_days"] = agg["total_orders"].apply(
        lambda o: max(1, int(np.random.normal(2, 0.5)))
        if not pd.isna(o) else np.nan
    )
    agg["on_time_delivery"] = agg["actual_lead_days"] <= agg["expected_lead_days"]

    # Issue #11: Fulfillment delay on high-volume days (should correlate; here it doesn't)
    high_vol_idx = agg[agg["total_orders"] > agg["total_orders"].quantile(0.90)].index
    agg.loc[high_vol_idx.intersection(
        agg.sample(frac=0.3, random_state=5).index
    ), "actual_lead_days"] = 1   # suspiciously fast on high-volume days → data error

    return agg


# ── Master runner ───────────────────────────────────────────────────────────
def generate_all(output_dir: str = "data") -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    print("Generating products …")
    products_df = generate_products()

    print("Generating orders …")
    orders_df = generate_orders()

    print("Generating sales …")
    sales_df = generate_sales(orders_df)

    print("Generating inventory …")
    inventory_df = generate_inventory(sales_df)

    print("Generating customers …")
    customers_df = generate_customers()

    print("Generating supply-chain …")
    supply_df = generate_supply_chain(orders_df)

    datasets = {
        "products":     products_df,
        "orders":       orders_df,
        "sales":        sales_df,
        "inventory":    inventory_df,
        "customers":    customers_df,
        "supply_chain": supply_df,
    }
    for name, df in datasets.items():
        path = os.path.join(output_dir, f"retail_{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {path}  ({len(df):,} rows × {len(df.columns)} cols)")

    print("\nDataset generation complete.")
    return datasets


if __name__ == "__main__":
    generate_all()
