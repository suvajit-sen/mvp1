"""
Contextual Anomaly Detection Tool
===================================
Goes beyond single-column outlier detection by evaluating anomalies
in the context of correlated business metrics.

Business scenarios covered
--------------------------
1. Sales-vs-Orders contextual anomaly  – high sales backed by high orders is NOT anomalous
2. Returns-vs-Sales ratio anomaly      – return spike on a normal-sales day IS anomalous
3. Inventory-vs-Sales mismatch         – stock-out on a high-demand day IS anomalous
4. Fulfillment speed vs order volume   – unusually fast lead time on peak days may be a data error
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
from typing import Any


def _zscore_flag(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """Return boolean mask of rows whose z-score exceeds *threshold*."""
    clean = series.dropna()
    if clean.empty:
        return pd.Series(False, index=series.index)
    z = np.abs(stats.zscore(clean))
    flags = pd.Series(False, index=series.index)
    flags[clean.index] = z > threshold
    return flags


def sales_vs_orders_anomaly(
    sales_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    z_threshold: float = 2.5,
) -> pd.DataFrame:
    """
    Detect days where sales_qty is a statistical outlier BUT is NOT backed by a
    similarly elevated order volume.  A true anomaly exists only when the order
    level is *normal* while sales spiked (possible data-entry error or fraud).

    Parameters
    ----------
    sales_df  : daily sales aggregated to (date, store_id, product_id)
    orders_df : daily orders (date, store_id, product_id, orders_received)

    Returns
    -------
    DataFrame of flagged rows with an 'anomaly_type' column.
    """
    agg_sales = (
        sales_df.groupby(["date", "store_id", "product_id"])
        .agg(sales_qty=("sales_qty", "sum"), sales_amount=("sales_amount", "sum"))
        .reset_index()
    )
    agg_orders = (
        orders_df.groupby(["date", "store_id", "product_id"])["orders_received"]
        .sum().reset_index()
    )
    merged = agg_sales.merge(agg_orders, on=["date", "store_id", "product_id"], how="left")

    results = []
    for (sid, pid), grp in merged.groupby(["store_id", "product_id"]):
        grp = grp.copy().sort_values("date")
        if len(grp) < 10:
            continue

        sales_outlier = _zscore_flag(grp["sales_qty"], z_threshold)
        orders_outlier = _zscore_flag(grp["orders_received"].fillna(grp["orders_received"].median()),
                                      z_threshold)

        # True anomaly: sales is an outlier but orders are NOT
        true_anomaly = sales_outlier & ~orders_outlier
        # Explained spike: both sales AND orders are high (legitimate)
        explained_spike = sales_outlier & orders_outlier

        flagged = grp[true_anomaly].copy()
        flagged["anomaly_type"] = "SALES_WITHOUT_ORDER_BACKING"
        flagged["sales_zscore"] = stats.zscore(grp["sales_qty"].fillna(0))[
            np.where(true_anomaly)[0]
        ]

        explained = grp[explained_spike].copy()
        explained["anomaly_type"] = "EXPLAINED_SPIKE_HIGH_ORDERS"
        explained["sales_zscore"] = stats.zscore(grp["sales_qty"].fillna(0))[
            np.where(explained_spike)[0]
        ]

        results.extend([flagged, explained])

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def returns_anomaly(sales_df: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    """
    Detect abnormal return rates: flag rows where returns / sales_qty ratio
    is an outlier AND the sales volume is not abnormally high (i.e. returns
    are not simply proportional to a legitimate sales spike).
    """
    agg = (
        sales_df.groupby(["date", "store_id", "product_id"])
        .agg(sales_qty=("sales_qty", "sum"), returns_qty=("returns_qty", "sum"))
        .reset_index()
    )
    agg["return_rate"] = agg["returns_qty"] / agg["sales_qty"].replace(0, np.nan)

    results = []
    for (sid, pid), grp in agg.groupby(["store_id", "product_id"]):
        grp = grp.copy().sort_values("date")
        if len(grp) < 10:
            continue

        rate_outlier = _zscore_flag(grp["return_rate"].fillna(0), z_threshold)
        sales_outlier = _zscore_flag(grp["sales_qty"], z_threshold)

        # Only flag if return_rate spike is NOT driven by a sales spike
        genuine_return_anomaly = rate_outlier & ~sales_outlier

        flagged = grp[genuine_return_anomaly].copy()
        flagged["anomaly_type"] = "RETURN_RATE_SPIKE_NOT_SALES_DRIVEN"
        results.append(flagged)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def inventory_accuracy_anomaly(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where closing_stock does not reconcile with
    opening_stock - units_sold + restocked_qty.
    Also flags stock-outs on days where demand was non-zero.

    Uses a 99th-percentile threshold on the reconciliation difference so that
    cascading artefacts from a single injected error do not inflate the count.
    """
    df = inventory_df.copy()
    df["expected_closing"] = df["opening_stock"] - df["units_sold"] + df["restocked_qty"]
    df["reconciliation_diff"] = (df["closing_stock"] - df["expected_closing"]).abs()

    # Only flag the worst outliers (top 1 %) to avoid cascade inflation
    threshold = df["reconciliation_diff"].quantile(0.99)
    ghost_inventory = df[df["reconciliation_diff"] > threshold].copy()
    ghost_inventory["anomaly_type"] = "INVENTORY_RECONCILIATION_ERROR"

    # Only flag stockouts if demand was non-zero AND it's truly unexpected
    # (units_sold == 0 because of stock-out while orders were coming in)
    stockouts = df[(df["closing_stock"] == 0) & (df["units_sold"] > 0) &
                   (df["restocked_qty"] == 0)].copy()
    stockouts["anomaly_type"] = "STOCKOUT_WITH_ACTIVE_DEMAND"

    return pd.concat([ghost_inventory, stockouts], ignore_index=True)


def fulfillment_anomaly(supply_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag fulfillment records where lead time is suspiciously fast on
    high-order-volume days (potential data entry error).
    """
    df = supply_df.copy()
    high_vol_threshold = df["total_orders"].quantile(0.85)
    fast_delivery_threshold = df["actual_lead_days"].quantile(0.10)

    anomalies = df[
        (df["total_orders"] >= high_vol_threshold) &
        (df["actual_lead_days"] <= fast_delivery_threshold)
    ].copy()
    anomalies["anomaly_type"] = "UNREALISTICALLY_FAST_FULFILLMENT_ON_PEAK_DAY"
    return anomalies


def contextual_anomaly_detection(
    sales_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    supply_df: pd.DataFrame,
    z_threshold: float = 2.5,
) -> dict[str, Any]:
    """
    Run all contextual anomaly detection checks and return a summary dict.
    """
    sales_order_anomalies = sales_vs_orders_anomaly(sales_df, orders_df, z_threshold)
    return_anomalies = returns_anomaly(sales_df, z_threshold)
    inv_anomalies = inventory_accuracy_anomaly(inventory_df)
    fulfillment_anomalies = fulfillment_anomaly(supply_df)

    true_anomalies = (
        sales_order_anomalies[
            sales_order_anomalies["anomaly_type"] == "SALES_WITHOUT_ORDER_BACKING"
        ] if not sales_order_anomalies.empty else pd.DataFrame()
    )
    explained_spikes = (
        sales_order_anomalies[
            sales_order_anomalies["anomaly_type"] == "EXPLAINED_SPIKE_HIGH_ORDERS"
        ] if not sales_order_anomalies.empty else pd.DataFrame()
    )

    return {
        "sales_order_true_anomalies":    true_anomalies,
        "sales_order_explained_spikes":  explained_spikes,
        "return_rate_anomalies":         return_anomalies,
        "inventory_anomalies":           inv_anomalies,
        "fulfillment_anomalies":         fulfillment_anomalies,
        "summary": {
            "sales_true_anomaly_count":    len(true_anomalies),
            "explained_spike_count":       len(explained_spikes),
            "return_anomaly_count":        len(return_anomalies),
            "inventory_anomaly_count":     len(inv_anomalies),
            "fulfillment_anomaly_count":   len(fulfillment_anomalies),
        },
    }
