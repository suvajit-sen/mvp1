"""
Statistical Tests Tool
========================
Applies hypothesis testing and distribution-based checks grounded in
business context to surface data quality issues.

Scenarios covered
-----------------
9.  Seasonal pattern violation  – month-over-month deviation from expected seasonal index
10. Churn risk paradox          – newly acquired Premium customers with high churn risk
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
from typing import Any

# Expected seasonal sales index (month → relative multiplier, baseline = 1.0)
EXPECTED_SEASONAL_INDEX: dict[int, float] = {
    1: 0.80, 2: 0.75, 3: 0.85, 4: 0.90, 5: 0.95, 6: 1.00,
    7: 1.05, 8: 1.10, 9: 1.00, 10: 1.05, 11: 1.30, 12: 1.60,
}


def seasonal_pattern_test(sales_df: pd.DataFrame) -> dict[str, Any]:
    """
    For each product-store pair, compare the actual monthly sales index against
    the expected seasonal index using a one-sample t-test.
    Significant deviations (p < 0.05) are flagged as seasonal pattern violations.
    """
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    monthly_sales = (
        df.groupby(["store_id", "product_id", "month"])["sales_qty"]
        .sum().reset_index()
    )

    results = []
    for (sid, pid), grp in monthly_sales.groupby(["store_id", "product_id"]):
        if len(grp) < 6:
            continue
        # Normalize actual sales to form an index
        mean_sales = grp["sales_qty"].mean()
        if mean_sales == 0:
            continue
        grp = grp.copy()
        grp["actual_index"] = grp["sales_qty"] / mean_sales
        grp["expected_index"] = grp["month"].map(EXPECTED_SEASONAL_INDEX)
        grp["deviation"] = grp["actual_index"] - grp["expected_index"]

        # One-sample t-test: is the mean deviation significantly different from 0?
        t_stat, p_val = stats.ttest_1samp(grp["deviation"].dropna(), popmean=0)

        for _, row in grp.iterrows():
            results.append({
                "store_id":       sid,
                "product_id":     pid,
                "month":          int(row["month"]),
                "actual_index":   round(row["actual_index"], 3),
                "expected_index": round(row["expected_index"], 3),
                "deviation":      round(row["deviation"], 3),
                "t_stat":         round(t_stat, 4),
                "p_value":        round(p_val, 4),
                "flagged":        (p_val < 0.05) and (abs(row["deviation"]) > 0.30),
            })

    df_results = pd.DataFrame(results)
    flagged = df_results[df_results["flagged"]] if not df_results.empty else df_results
    return {
        "seasonal_test_table": df_results,
        "flagged_violations":  flagged,
        "summary": f"{len(flagged)} store-product-month combinations violate the expected seasonal pattern.",
    }


def churn_risk_paradox_test(customers_df: pd.DataFrame) -> dict[str, Any]:
    """
    Newly acquired Premium customers with high churn risk (> 0.85) are
    statistically anomalous.  Use a chi-square test to check if the
    distribution of high churn risk differs significantly between
    new vs established Premium customers.
    """
    df = customers_df.copy()
    premium = df[df["segment"] == "Premium"].copy()
    premium["is_new"] = premium["acquisition_month"] >= 10
    premium["high_churn"] = premium["churn_risk_score"] > 0.85

    contingency = pd.crosstab(premium["is_new"], premium["high_churn"])
    if contingency.shape == (2, 2):
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    else:
        chi2, p_val, dof = 0.0, 1.0, 0

    flagged = premium[premium["is_new"] & premium["high_churn"]].copy()
    flagged["issue"] = "NEW_PREMIUM_CUSTOMER_HIGH_CHURN_RISK"

    return {
        "contingency_table": contingency,
        "chi2_stat":         round(chi2, 4),
        "p_value":           round(p_val, 4),
        "dof":               dof,
        "flagged_customers": flagged,
        "summary": (
            f"Chi-square test p={p_val:.4f}. "
            f"{len(flagged)} new Premium customers have suspiciously high churn risk (>0.85)."
        ),
    }


def fulfillment_lead_time_test(supply_df: pd.DataFrame) -> dict[str, Any]:
    """
    Kolmogorov-Smirnov test: compare the lead-time distribution on peak days
    vs normal days.  If peak days show a significantly *shorter* lead time,
    it may indicate data recording errors.
    """
    df = supply_df.dropna(subset=["actual_lead_days", "total_orders"]).copy()
    high_vol = df["total_orders"] >= df["total_orders"].quantile(0.85)
    peak_lead   = df[high_vol]["actual_lead_days"]
    normal_lead = df[~high_vol]["actual_lead_days"]

    ks_stat, p_val = stats.ks_2samp(peak_lead, normal_lead)
    peak_faster = peak_lead.mean() < normal_lead.mean()

    return {
        "peak_mean_lead_days":   round(peak_lead.mean(), 3),
        "normal_mean_lead_days": round(normal_lead.mean(), 3),
        "ks_stat":               round(ks_stat, 4),
        "p_value":               round(p_val, 4),
        "anomaly_detected":      p_val < 0.05 and peak_faster,
        "summary": (
            f"KS test p={p_val:.4f}. "
            + (
                "ANOMALY: Peak-day lead times are paradoxically faster than normal days."
                if (p_val < 0.05 and peak_faster)
                else "Lead-time distribution appears consistent across volume levels."
            )
        ),
    }


def run_statistical_tests(
    sales_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    supply_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run all statistical business data quality tests.
    """
    seasonal_result    = seasonal_pattern_test(sales_df)
    churn_result       = churn_risk_paradox_test(customers_df)
    fulfillment_result = fulfillment_lead_time_test(supply_df)

    return {
        "seasonal_patterns":    seasonal_result,
        "churn_risk_paradox":   churn_result,
        "fulfillment_lead_time": fulfillment_result,
        "overall_summary": {
            "seasonal_violations":  len(seasonal_result["flagged_violations"]),
            "churn_paradox_count":  len(churn_result["flagged_customers"]),
            "fulfillment_anomaly":  fulfillment_result["anomaly_detected"],
        },
    }
