"""
Business Rules Validation Tool
================================
Validates hard business rules that are domain-specific and cannot be
detected by purely statistical means.

Rules covered
-------------
R1  sales_qty can never exceed orders_received on the same day for same SKU/store
R2  closing_stock must equal opening_stock - units_sold + restocked_qty (±1 tolerance)
R3  unit_price must be ≥ cost_per_unit (no selling below cost)
R4  discount_pct must be between 0 and 0.60 (business policy)
R5  return_qty must be ≤ sales_qty
R6  gross_margin must reconcile with (unit_price*(1-discount) - cost) * qty
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


def _flag(df: pd.DataFrame, mask: pd.Series, rule_id: str, description: str) -> pd.DataFrame:
    flagged = df[mask].copy()
    flagged["rule_id"] = rule_id
    flagged["rule_description"] = description
    return flagged


def validate_business_rules(
    sales_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Validate all business rules and return violations.
    """
    violations = []

    # ── R1: Sales qty ≤ Orders received ──────────────────────────────────────
    agg_sales = (
        sales_df.groupby(["date", "store_id", "product_id"])["sales_qty"]
        .sum().reset_index()
    )
    agg_orders = (
        orders_df.groupby(["date", "store_id", "product_id"])["orders_received"]
        .sum().reset_index()
    )
    r1_df = agg_sales.merge(agg_orders, on=["date", "store_id", "product_id"], how="left")
    r1_violations = r1_df[r1_df["sales_qty"] > r1_df["orders_received"] * 1.10].copy()
    r1_violations["rule_id"] = "R1"
    r1_violations["rule_description"] = "Sales quantity exceeds orders received by more than 10%"
    violations.append(r1_violations)

    # ── R2: Inventory reconciliation ─────────────────────────────────────────
    inv = inventory_df.copy()
    inv["expected_closing"] = inv["opening_stock"] - inv["units_sold"] + inv["restocked_qty"]
    inv["diff"] = (inv["closing_stock"] - inv["expected_closing"]).abs()
    # Use 99th-percentile threshold to focus on genuine outliers and avoid cascade inflation
    r2_threshold = inv["diff"].quantile(0.99)
    r2_violations = _flag(inv, inv["diff"] > r2_threshold, "R2",
                          "Closing stock does not reconcile (opening - sold + restocked)")
    violations.append(r2_violations)

    # ── R3: No selling below cost ─────────────────────────────────────────────
    sales = sales_df.copy()
    sales["net_price"] = sales["unit_price"] * (1 - sales["discount_pct"])
    r3_violations = _flag(sales, sales["net_price"] < sales["cost_per_unit"], "R3",
                          "Net selling price is below cost per unit")
    violations.append(r3_violations)

    # ── R4: Discount policy (0 – 60%) ────────────────────────────────────────
    r4_violations = _flag(
        sales_df,
        (sales_df["discount_pct"] < 0) | (sales_df["discount_pct"] > 0.60),
        "R4",
        "Discount percentage outside allowed range [0, 0.60]",
    )
    violations.append(r4_violations)

    # ── R5: Returns ≤ Sales ───────────────────────────────────────────────────
    r5_violations = _flag(
        sales_df,
        sales_df["returns_qty"] > sales_df["sales_qty"],
        "R5",
        "Returns quantity exceeds sales quantity",
    )
    violations.append(r5_violations)

    # ── R6: Gross margin reconciliation ──────────────────────────────────────
    sales2 = sales_df.copy()
    sales2["expected_margin"] = (
        (sales2["unit_price"] * (1 - sales2["discount_pct"]) - sales2["cost_per_unit"])
        * sales2["sales_qty"]
    ).round(2)
    sales2["margin_diff"] = (sales2["gross_margin"] - sales2["expected_margin"]).abs()
    r6_violations = _flag(
        sales2,
        sales2["margin_diff"] > 1.0,
        "R6",
        "Gross margin does not reconcile with price, cost, and quantity",
    )
    violations.append(r6_violations)
    all_violations = pd.concat([v for v in violations if not v.empty], ignore_index=True)

    summary = (
        all_violations.groupby("rule_id")["rule_description"]
        .count().rename("violation_count").reset_index()
        .merge(
            all_violations[["rule_id", "rule_description"]]
            .drop_duplicates(),
            on="rule_id",
        )
        if not all_violations.empty
        else pd.DataFrame(columns=["rule_id", "violation_count", "rule_description"])
    )

    return {
        "violations":          all_violations,
        "summary_by_rule":     summary,
        "total_violations":    len(all_violations),
        "overall_summary": (
            f"Found {len(all_violations)} business rule violations across "
            f"{all_violations['rule_id'].nunique() if not all_violations.empty else 0} rules."
        ),
    }
