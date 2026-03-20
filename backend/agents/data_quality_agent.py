"""
Business Data Quality Agent – LangGraph Orchestrator
======================================================
Builds a LangGraph state-machine that routes analysis requests to the
appropriate data-quality tools and synthesises findings into a report.

Graph topology
--------------

  ┌─────────┐
  │  START  │
  └────┬────┘
       │
       ▼
  ┌──────────┐     decides which tools to run
  │  router  │────────────────────────────────────────────┐
  └──────────┘                                            │
       │                                                  │
       ▼                                                  │
  ┌────────────────────────────────────────────────────┐  │
  │ parallel tool nodes (run only those selected):    │  │
  │   anomaly_detector  │  correlation_analyzer        │  │
  │   statistical_tester│  business_rule_validator     │  │
  └────────────┬───────────────────────────────────────┘  │
               │                                          │
               ▼                                          │
         ┌───────────────┐                                │
         │report_generator│◄───────────────────────────────┘
         └───────┬────────┘
                 │
                 ▼
              ┌─────┐
              │ END │
              └─────┘

The graph can be used with a real LLM (OpenAI / Anthropic / etc.) by
setting the OPENAI_API_KEY environment variable, or without any LLM by
using the built-in heuristic router (set USE_LLM=False).
"""

from __future__ import annotations

import os
from typing import Any, Literal

import pandas as pd
from typing_extensions import TypedDict

# LangGraph
from langgraph.graph import StateGraph, END

# Local tools
from .tools.anomaly_detection import contextual_anomaly_detection
from .tools.correlation_analysis import run_correlation_analysis
from .tools.statistical_tests import run_statistical_tests
from .tools.business_rules import validate_business_rules


# ── State definition ────────────────────────────────────────────────────────

class DQState(TypedDict, total=False):
    """Shared state passed through the LangGraph nodes."""
    # Input datasets
    sales_df:     pd.DataFrame
    orders_df:    pd.DataFrame
    inventory_df: pd.DataFrame
    customers_df: pd.DataFrame
    supply_df:    pd.DataFrame

    # Which analyses to run (filled by router)
    run_anomaly:     bool
    run_correlation: bool
    run_statistical: bool
    run_rules:       bool

    # Results (filled by tool nodes)
    anomaly_results:     dict[str, Any]
    correlation_results: dict[str, Any]
    statistical_results: dict[str, Any]
    rules_results:       dict[str, Any]

    # Final output
    final_report: str
    llm_narrative: str


# ── Node implementations ─────────────────────────────────────────────────────

def router_node(state: DQState) -> DQState:
    """
    Heuristic router: enables all four analysis categories.
    When an LLM is available (OPENAI_API_KEY set), this could be replaced
    with a prompt-driven router that picks analyses based on a user question.

    NOTE: only return the *new* keys this node writes so that parallel
    downstream nodes do not conflict on shared input-data keys.
    """
    return {
        "run_anomaly":     True,
        "run_correlation": True,
        "run_statistical": True,
        "run_rules":       True,
    }


def anomaly_node(state: DQState) -> DQState:
    if not state.get("run_anomaly", False):
        return {}
    results = contextual_anomaly_detection(
        sales_df=state["sales_df"],
        orders_df=state["orders_df"],
        inventory_df=state["inventory_df"],
        supply_df=state["supply_df"],
    )
    return {"anomaly_results": results}


def correlation_node(state: DQState) -> DQState:
    if not state.get("run_correlation", False):
        return {}
    results = run_correlation_analysis(
        sales_df=state["sales_df"],
        customers_df=state["customers_df"],
    )
    return {"correlation_results": results}


def statistical_node(state: DQState) -> DQState:
    if not state.get("run_statistical", False):
        return {}
    results = run_statistical_tests(
        sales_df=state["sales_df"],
        customers_df=state["customers_df"],
        supply_df=state["supply_df"],
    )
    return {"statistical_results": results}


def rules_node(state: DQState) -> DQState:
    if not state.get("run_rules", False):
        return {}
    results = validate_business_rules(
        sales_df=state["sales_df"],
        orders_df=state["orders_df"],
        inventory_df=state["inventory_df"],
    )
    return {"rules_results": results}


def report_node(state: DQState) -> DQState:
    """Synthesises all findings into a structured text report."""
    sections: list[str] = ["=" * 70, "BUSINESS DATA QUALITY REPORT", "=" * 70]

    # ── Anomaly section ───────────────────────────────────────────────────
    if (anom := state.get("anomaly_results")):
        s = anom.get("summary", {})
        sections.append("\n[1] CONTEXTUAL ANOMALY DETECTION")
        sections.append(
            f"  • Sales-without-order backing anomalies : {s.get('sales_true_anomaly_count', 0)}"
        )
        sections.append(
            f"  • Explained spikes (high orders ↔ high sales): {s.get('explained_spike_count', 0)}"
        )
        sections.append(
            f"  • Return-rate anomalies                 : {s.get('return_anomaly_count', 0)}"
        )
        sections.append(
            f"  • Inventory reconciliation issues       : {s.get('inventory_anomaly_count', 0)}"
        )
        sections.append(
            f"  • Fulfillment speed anomalies           : {s.get('fulfillment_anomaly_count', 0)}"
        )

    # ── Correlation section ───────────────────────────────────────────────
    if (corr := state.get("correlation_results")):
        s = corr.get("overall_summary", {})
        sections.append("\n[2] CORRELATION-BASED CHECKS")
        sections.append(f"  • Discount-with-no-volume-gain pairs : {s.get('discount_issues', 0)}")
        sections.append(f"  • Price inconsistency across stores  : {s.get('price_issues', 0)}")
        sections.append(f"  • Rows sold below cost               : {s.get('margin_issues', 0)}")
        sections.append(f"  • CLV / AOV mismatches               : {s.get('clv_aov_issues', 0)}")

    # ── Statistical section ───────────────────────────────────────────────
    if (stat := state.get("statistical_results")):
        s = stat.get("overall_summary", {})
        sections.append("\n[3] STATISTICAL TESTS")
        sections.append(
            f"  • Seasonal pattern violations        : {s.get('seasonal_violations', 0)}"
        )
        sections.append(
            f"  • New-Premium high-churn paradox     : {s.get('churn_paradox_count', 0)}"
        )
        sections.append(
            f"  • Fulfillment lead-time anomaly      : {'YES' if s.get('fulfillment_anomaly') else 'NO'}"
        )

    # ── Business rules section ────────────────────────────────────────────
    if (rules := state.get("rules_results")):
        sections.append("\n[4] BUSINESS RULE VIOLATIONS")
        sections.append(f"  {rules.get('overall_summary', '')}")
        summary_df = rules.get("summary_by_rule", pd.DataFrame())
        if not summary_df.empty:
            for _, row in summary_df.iterrows():
                sections.append(
                    f"  • {row['rule_id']}: {row['rule_description']} "
                    f"[{row['violation_count']} violations]"
                )

    sections.append("\n" + "=" * 70)
    report = "\n".join(sections)

    # Optionally enrich with LLM narrative
    llm_narrative = _generate_llm_narrative(state, report)

    return {"final_report": report, "llm_narrative": llm_narrative}


def _generate_llm_narrative(state: DQState, summary_report: str) -> str:
    """
    Call an LLM to produce a plain-English narrative and prioritised action
    list.  Falls back gracefully when no API key is available.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "[LLM narrative not available – set OPENAI_API_KEY to enable AI-powered "
            "plain-English summaries and prioritised remediation recommendations.]"
        )

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        messages = [
            SystemMessage(content=(
                "You are a senior data scientist and business analyst. "
                "Given a structured data-quality report, produce: "
                "1) a plain-English executive summary (3-5 sentences), "
                "2) the top 5 prioritised action items with business impact."
            )),
            HumanMessage(content=summary_report),
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as exc:  # pragma: no cover
        return f"[LLM narrative generation failed: {exc}]"


# ── Graph builder ────────────────────────────────────────────────────────────

def build_dq_graph() -> StateGraph:
    """Build and compile the LangGraph data-quality workflow."""
    graph = StateGraph(DQState)

    graph.add_node("router",       router_node)
    graph.add_node("anomaly",      anomaly_node)
    graph.add_node("correlation",  correlation_node)
    graph.add_node("statistical",  statistical_node)
    graph.add_node("rules",        rules_node)
    graph.add_node("report",       report_node)

    graph.set_entry_point("router")

    # Router fans out to all four analysis nodes
    graph.add_edge("router",      "anomaly")
    graph.add_edge("router",      "correlation")
    graph.add_edge("router",      "statistical")
    graph.add_edge("router",      "rules")

    # All four nodes converge on the report node
    graph.add_edge("anomaly",     "report")
    graph.add_edge("correlation", "report")
    graph.add_edge("statistical", "report")
    graph.add_edge("rules",       "report")

    graph.add_edge("report", END)

    return graph.compile()


# ── Public API ───────────────────────────────────────────────────────────────

def run_dq_analysis(
    sales_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    supply_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Execute the full business data-quality pipeline and return the state dict
    containing all results and the final report.
    """
    app = build_dq_graph()
    initial_state: DQState = {
        "sales_df":     sales_df,
        "orders_df":    orders_df,
        "inventory_df": inventory_df,
        "customers_df": customers_df,
        "supply_df":    supply_df,
    }
    final_state = app.invoke(initial_state)
    return final_state
