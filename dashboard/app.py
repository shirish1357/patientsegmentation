"""Plotly Dash dashboard — Medicare Beneficiary Segmentation.

Launch: python dashboard/app.py
Then open http://localhost:8050 in a browser.

Requires data/processed/segments.parquet (produced by notebooks/01_pipeline.ipynb).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dash_table, dcc, html

SEGMENTS_PATH = Path("data/processed/segments.parquet")

app = Dash(__name__, title="Medicare Patient Segments")

# ── Layout ─────────────────────────────────────────────────────────────────

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, sans-serif", "maxWidth": "1280px",
           "margin": "0 auto", "padding": "24px"},
    children=[
        html.H1("Medicare Beneficiary Segmentation",
                style={"color": "#1a1a2e", "marginBottom": "4px"}),
        html.P(
            "CMS DE-SynPUF Sample 1 · 2009 baseline · 2010 outcomes · Chronic-condition cohort",
            style={"color": "#666", "marginBottom": "24px"},
        ),

        # Algorithm toggle
        html.Div([
            html.Label("Algorithm:", style={"fontWeight": "600", "marginRight": "8px"}),
            dcc.RadioItems(
                id="algo-toggle",
                options=[
                    {"label": "  K-Means", "value": "kmeans"},
                    {"label": "  GMM", "value": "gmm"},
                ],
                value="kmeans",
                inline=True,
                style={"fontSize": "15px"},
            ),
        ], style={"marginBottom": "32px"}),

        # Row 1: Overview bar + Radar
        html.Div([
            html.Div(dcc.Graph(id="bar-overview"),
                     style={"flex": "1", "minWidth": "0"}),
            html.Div(dcc.Graph(id="radar-features"),
                     style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "24px", "marginBottom": "32px"}),

        # Row 2: Drill-down segment selector + detail table
        html.Div([
            html.Label("Select segment to drill down:",
                       style={"fontWeight": "600", "marginBottom": "8px"}),
            dcc.Dropdown(id="segment-dropdown", clearable=False,
                         style={"marginBottom": "16px"}),
            html.Div(id="detail-table"),
        ], style={"marginBottom": "32px"}),

        # Prioritization framework table
        html.H2("Prioritization Framework",
                style={"color": "#1a1a2e", "marginBottom": "12px"}),
        html.Div(id="priority-table"),

        html.Hr(),
        html.P(
            "⚠ DE-SynPUF data is synthetic and not clinically validated. "
            "Patterns are plausible but should not be used for clinical decisions.",
            style={"color": "#888", "fontSize": "13px"},
        ),
    ],
)


# ── Data loading ────────────────────────────────────────────────────────────

def _load_segments(algorithm: str) -> pd.DataFrame:
    if not SEGMENTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(SEGMENTS_PATH)
    if "algorithm" in df.columns:
        df = df[df["algorithm"] == algorithm]
    return df


# ── Callbacks ───────────────────────────────────────────────────────────────

@callback(
    Output("bar-overview", "figure"),
    Output("radar-features", "figure"),
    Output("segment-dropdown", "options"),
    Output("segment-dropdown", "value"),
    Output("priority-table", "children"),
    Input("algo-toggle", "value"),
)
def update_charts(algorithm: str):
    df = _load_segments(algorithm)

    if df.empty:
        empty_fig = go.Figure().add_annotation(
            text="Run the pipeline first (notebooks/01_pipeline.ipynb)",
            showarrow=False, font={"size": 16}
        )
        return empty_fig, empty_fig, [], None, "No data — run pipeline first."

    seg_name_col = "segment_name" if "segment_name" in df.columns else "cluster"
    df["label"] = df[seg_name_col].astype(str)

    # ── Bar chart: mean 2010 cost + cohort size ──────────────────────────
    cost_col = next((c for c in ["mean_cost_2010", "mean_total_cost_2010"] if c in df.columns), None)
    if cost_col:
        bar_df = df[["label", "n", cost_col]].copy()
        bar_df = bar_df.sort_values(cost_col, ascending=False)
        bar_fig = px.bar(
            bar_df, x="label", y=cost_col,
            text="n",
            color=cost_col,
            color_continuous_scale="Blues",
            labels={"label": "Segment", cost_col: "Mean 2010 Total Cost ($)", "n": "N"},
            title="Segment Size & Mean 2010 Cost",
        )
        bar_fig.update_traces(texttemplate="%{text:,.0f} beneficiaries",
                              textposition="outside")
        bar_fig.update_layout(coloraxis_showscale=False,
                              margin={"t": 50, "b": 80},
                              plot_bgcolor="white")
    else:
        bar_fig = go.Figure().add_annotation(text="mean_total_cost_2010 not found",
                                             showarrow=False)

    # ── Radar chart: standardized clinical + utilization features ────────
    radar_cols = {
        "Charlson": "mean_charlson",
        "# Chronic": "mean_n_chronic",
        "Polypharmacy": "mean_polypharmacy",
        "IP Admits": "mean_ip_admissions_2009",
        "ED Visits": "mean_ed_visits_2009",
        "OP Visits": "mean_op_visits_2009",
    }
    present = {k: v for k, v in radar_cols.items() if v in df.columns}

    if present:
        radar_df = df[["label"] + list(present.values())].set_index("label")
        # Min-max scale each feature across segments
        scaled = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)
        radar_fig = go.Figure()
        for seg_label, row in scaled.iterrows():
            radar_fig.add_trace(go.Scatterpolar(
                r=row.values.tolist() + [row.values[0]],
                theta=list(present.keys()) + [list(present.keys())[0]],
                fill="toself", opacity=0.5, name=seg_label,
            ))
        radar_fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            title="Segment Profiles (Normalized Features)",
            margin={"t": 50, "b": 60, "l": 80, "r": 80},
        )
    else:
        radar_fig = go.Figure().add_annotation(text="Profile features not found",
                                               showarrow=False)

    # ── Dropdown options ─────────────────────────────────────────────────
    options = [{"label": row["label"], "value": row["cluster"]} for _, row in df.iterrows()]
    default_value = df["cluster"].iloc[0] if not df.empty else None

    # ── Priority table ───────────────────────────────────────────────────
    priority_cols = ["label", "n", "pct_of_cohort", "cost_tier", "modifiability",
                     "outreach_intensity", "intervention_type"]
    prio_present = [c for c in priority_cols if c in df.columns]
    if len(prio_present) >= 3:
        prio_table = dash_table.DataTable(
            data=df[prio_present].rename(columns={
                "label": "Segment", "n": "N", "pct_of_cohort": "% Cohort",
                "cost_tier": "Cost Tier", "modifiability": "Modifiability",
                "outreach_intensity": "Outreach Intensity",
                "intervention_type": "Intervention",
            }).to_dict("records"),
            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px",
                        "maxWidth": "300px", "whiteSpace": "normal"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f0f4ff"},
            style_data_conditional=[
                {"if": {"filter_query": "{Outreach Intensity} = 'Intensive'"},
                 "backgroundColor": "#fff0f0"},
                {"if": {"filter_query": "{Outreach Intensity} = 'Moderate'"},
                 "backgroundColor": "#fff8e8"},
            ],
        )
    else:
        prio_table = html.P("Prioritization columns not found — run pipeline.")

    return bar_fig, radar_fig, options, default_value, prio_table


@callback(
    Output("detail-table", "children"),
    Input("segment-dropdown", "value"),
    Input("algo-toggle", "value"),
)
def update_detail(cluster_id, algorithm: str):
    if cluster_id is None:
        return html.P("Select a segment above.")
    df = _load_segments(algorithm)
    if df.empty:
        return html.P("No data.")

    row = df[df["cluster"] == cluster_id]
    if row.empty:
        return html.P("Segment not found.")
    row = row.iloc[0]

    seg_name = row.get("segment_name", f"Segment {cluster_id}")
    top5 = row.get("top5_conditions", "N/A")
    intervention = row.get("intervention_type", "N/A")
    cms_codes = row.get("cms_codes", "N/A")
    jaccard = row.get("mean_jaccard")
    stab_tier = row.get("stability_tier", "N/A")

    def _fmt_cost(v):
        return f"${v:,.0f}" if v is not None and not np.isnan(float(v)) else "N/A"

    def _fmt_float(v, fmt=".2f", suffix=""):
        if v is None:
            return "N/A"
        try:
            f = float(v)
            return "N/A" if np.isnan(f) else f"{f:{fmt}}{suffix}"
        except (TypeError, ValueError):
            return "N/A"

    items = [
        ("Segment", seg_name),
        ("Beneficiaries (N)", f"{int(row.get('n', 0)):,}  ({row.get('pct_of_cohort', 0):.1f}% of cohort)"),
        ("Mean 2010 Total Cost", _fmt_cost(row.get("mean_cost_2010", row.get("mean_total_cost_2010")))),
        ("Median 2010 Total Cost", _fmt_cost(row.get("median_total_cost_2010"))),
        ("Mean IP Admits (2010)", _fmt_float(row.get("mean_ip_admissions_2010"))),
        ("Mean ED Visits (2010)", _fmt_float(row.get("mean_ed_visits_2010"))),
        ("30-day Readmit Rate", _fmt_float(row.get("mean_readmit_30d_2010"), fmt=".1%")),
        ("Mean Charlson Index", _fmt_float(row.get("mean_charlson"))),
        ("Mean # Chronic Conditions", _fmt_float(row.get("mean_n_chronic"), fmt=".1f")),
        ("Mean Age", _fmt_float(row.get("mean_age"), fmt=".1f", suffix=" years")),
        ("% Female", _fmt_float(row.get("pct_female"), fmt=".1f", suffix="%")),
        ("Stability (mean Jaccard)", f"{jaccard:.2f}" if jaccard is not None else "N/A"),
        ("Stability Tier", stab_tier),
        ("Top 5 Conditions", top5),
        ("Recommended Intervention", intervention),
        ("CMS Billing Codes", cms_codes),
    ]

    return html.Table(
        [html.Tr([
            html.Td(k, style={"fontWeight": "600", "padding": "6px 12px",
                               "backgroundColor": "#f8f9fa", "width": "220px",
                               "verticalAlign": "top"}),
            html.Td(v, style={"padding": "6px 12px"}),
        ]) for k, v in items],
        style={"borderCollapse": "collapse", "width": "100%", "fontSize": "14px"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
