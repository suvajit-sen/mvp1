"""
Correlation Analysis Tool
==========================
Evaluates business-meaningful correlations between KPIs to surface
data quality issues that only become visible when metrics are compared.

Scenarios covered
-----------------
5.  Discount vs Revenue uplift        – deep discount should lift volume; if not, flag it
6.  Price consistency across stores   – same product should have similar price bands
7.  Sales vs Gross Margin correlation  – negative margin rows indicate pricing / cost errors
8.  Customer CLV vs Average Order Value – CLV should correlate with AOV; outliers are suspect
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
from typing import Any


def discount_revenue_correlation(sales_df: pd.DataFrame) -> dict[str, Any]:
    """
    For each (store, product) pair, compute Pearson correlation between
    discount_pct and sales_qty.  A healthy business expects a positive
    correlation (higher discount → more units sold).  Negative or near-zero
    correlation with high discount levels is a data quality / business anomaly.
    """
    results = []
    for (sid, pid), grp in sales_df.groupby(["store_id", "product_id"]):
        grp = grp.dropna(subset=["discount_pct", "sales_qty"])
        if len(grp) < 20:
            continue
        r, p_value = stats.pearsonr(grp["discount_pct"], grp["sales_qty"])
        avg_discount = grp["discount_pct"].mean()
        results.append({
            "store_id":    sid,
            "product_id":  pid,
            "pearson_r":   round(r, 4),
            "p_value":     round(p_value, 4),
            "avg_discount": round(avg_discount, 4),
            "flagged":     (r <= 0.10) and (avg_discount > 0.12),
            "issue":       "HIGH_DISCOUNT_NO_VOLUME_GAIN" if (r <= 0.10 and avg_discount > 0.12) else None,
        })
    df = pd.DataFrame(results)
    flagged = df[df["flagged"]] if not df.empty else df
    return {
        "correlation_table": df,
        "flagged_pairs":     flagged,
        "summary": f"{len(flagged)} store-product pairs show high discount with no volume gain.",
    }


def price_consistency_across_stores(sales_df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect products whose unit_price deviates significantly (> 2 std) from
    the cross-store mean.  Legitimate price variations are small; large
    deviations indicate pricing data errors.
    """
    price_stats = (
        sales_df.groupby(["product_id", "store_id"])["unit_price"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_price", "std": "price_std", "count": "n_txns"})
    )
    product_mean = (
        price_stats.groupby("product_id")["avg_price"]
        .mean().rename("product_avg_price").reset_index()
    )
    product_std = (
        price_stats.groupby("product_id")["avg_price"]
        .std().rename("product_price_std").reset_index()
    )
    merged = price_stats.merge(product_mean, on="product_id").merge(product_std, on="product_id")
    merged["z_price"] = (
        (merged["avg_price"] - merged["product_avg_price"])
        / merged["product_price_std"].replace(0, np.nan)
    ).round(3)
    flagged = merged[merged["z_price"].abs() > 1.5].copy()
    flagged["issue"] = "PRICE_INCONSISTENCY_ACROSS_STORES"
    return {
        "price_table":   merged,
        "flagged_stores": flagged,
        "summary": f"{len(flagged)} store-product combinations have anomalous pricing.",
    }


def sales_margin_correlation(sales_df: pd.DataFrame) -> dict[str, Any]:
    """
    Identify rows where gross_margin is negative, which suggests the
    cost_per_unit is above unit_price (selling below cost) – a serious
    business data / operational quality issue.
    """
    df = sales_df.copy()
    df["unit_margin"] = df["unit_price"] * (1 - df["discount_pct"]) - df["cost_per_unit"]
    negative_margin = df[df["unit_margin"] < 0].copy()
    negative_margin["issue"] = "SELLING_BELOW_COST"

    # Correlation between sales_qty and gross_margin by product
    corr_rows = []
    for pid, grp in df.groupby("product_id"):
        if len(grp) < 10:
            continue
        r, p = stats.pearsonr(grp["sales_qty"].fillna(0), grp["gross_margin"].fillna(0))
        corr_rows.append({"product_id": pid, "sales_margin_corr": round(r, 4), "p_value": round(p, 4)})

    return {
        "negative_margin_rows":  negative_margin,
        "margin_correlation":    pd.DataFrame(corr_rows),
        "summary": f"{len(negative_margin)} rows where products are sold below cost.",
    }


def clv_aov_consistency(customers_df: pd.DataFrame) -> dict[str, Any]:
    """
    CLV and Average Order Value should be positively correlated within each
    segment.  Premium customers with very low AOV are suspect records:
    their CLV implies high lifetime spend, but their per-order amount is
    inexplicably low compared to the rest of the segment.
    """
    df = customers_df.copy()
    results = []
    for seg, grp in df.groupby("segment"):
        r, p = stats.pearsonr(grp["clv_estimate"], grp["avg_order_value"])
        results.append({"segment": seg, "clv_aov_corr": round(r, 4), "p_value": round(p, 4)})

    # Flag Premium customers where AOV is abnormally low for their own segment
    # (below the 10th percentile of the Premium segment itself)
    premium_aov_p10 = df[df["segment"] == "Premium"]["avg_order_value"].quantile(0.10)
    premium_clv_median = df[df["segment"] == "Premium"]["clv_estimate"].median()

    df["clv_aov_ratio"] = df["clv_estimate"] / df["avg_order_value"].replace(0, np.nan)
    flagged = df[
        (df["segment"] == "Premium") &
        (df["avg_order_value"] < premium_aov_p10) &
        (df["clv_estimate"] > premium_clv_median * 0.5)   # still a real Premium CLV
    ].copy()
    flagged["issue"] = "CLV_AOV_INCONSISTENCY"

    return {
        "segment_correlations": pd.DataFrame(results),
        "flagged_customers":    flagged,
        "premium_aov_p10":      round(premium_aov_p10, 2),
        "summary": (
            f"{len(flagged)} Premium customers have AOV below the 10th percentile "
            f"of their own segment ({premium_aov_p10:.2f}) despite a significant CLV."
        ),
    }


def run_correlation_analysis(
    sales_df: pd.DataFrame,
    customers_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run all correlation-based business data quality checks.
    """
    discount_result  = discount_revenue_correlation(sales_df)
    price_result     = price_consistency_across_stores(sales_df)
    margin_result    = sales_margin_correlation(sales_df)
    clv_result       = clv_aov_consistency(customers_df)

    return {
        "discount_revenue":   discount_result,
        "price_consistency":  price_result,
        "sales_margin":       margin_result,
        "clv_aov":            clv_result,
        "overall_summary": {
            "discount_issues":  len(discount_result["flagged_pairs"]),
            "price_issues":     len(price_result["flagged_stores"]),
            "margin_issues":    len(margin_result["negative_margin_rows"]),
            "clv_aov_issues":   len(clv_result["flagged_customers"]),
        },
    }
