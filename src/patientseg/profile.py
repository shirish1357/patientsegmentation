"""Segment profiling: generate the segments.parquet table used by the briefing and dashboard."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CCW_FLAGS = [
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
    "SP_RA_OA", "SP_STRKETIA",
]

CCW_LABELS = {
    "SP_ALZHDMTA": "Alzheimer's / Dementia",
    "SP_CHF": "Congestive Heart Failure",
    "SP_CHRNKIDN": "Chronic Kidney Disease",
    "SP_CNCR": "Cancer",
    "SP_COPD": "COPD",
    "SP_DEPRESSN": "Depression",
    "SP_DIABETES": "Diabetes",
    "SP_ISCHMCHT": "Ischemic Heart Disease",
    "SP_OSTEOPRS": "Osteoporosis",
    "SP_RA_OA": "Rheumatoid / Osteoarthritis",
    "SP_STRKETIA": "Stroke / TIA",
}


def profile_segments(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    stability_df: pd.DataFrame,
    algorithm: str,
) -> pd.DataFrame:
    """Build per-segment profile table.

    Parameters
    ----------
    feature_df   : full feature matrix with outcome columns appended
    labels       : cluster labels aligned to feature_df rows
    stability_df : from stability.cluster_stability()
    algorithm    : "kmeans" or "gmm"

    Returns
    -------
    DataFrame with one row per segment.
    """
    df = feature_df.copy()
    df["cluster"] = labels
    df["algorithm"] = algorithm

    outcome_cols = ["total_cost_2010", "ip_admissions_2010", "ed_visits_2010", "readmit_30d_2010"]
    present_ccw = [f for f in CCW_FLAGS if f in df.columns]

    rows = []
    total_n = len(df)

    for c in sorted(df["cluster"].unique()):
        seg = df[df["cluster"] == c]
        n = len(seg)
        row: dict = {
            "algorithm": algorithm,
            "cluster": int(c),
            "n": n,
            "pct_of_cohort": round(100 * n / total_n, 1),
        }

        # 2010 outcomes
        for col in outcome_cols:
            if col in seg.columns:
                row[f"mean_{col}"] = round(seg[col].mean(), 2)
                if col == "total_cost_2010":
                    row["median_total_cost_2010"] = round(seg[col].median(), 2)

        # Clinical complexity
        row["mean_charlson"] = round(seg["charlson_index"].mean(), 2) if "charlson_index" in seg.columns else None
        row["mean_n_chronic"] = round(seg["n_chronic_conditions"].mean(), 2) if "n_chronic_conditions" in seg.columns else None
        row["mean_polypharmacy"] = round(seg["n_unique_drugs_2009"].mean(), 2) if "n_unique_drugs_2009" in seg.columns else None
        row["pct_esrd"] = round(100 * seg["esrd_indicator"].mean(), 1) if "esrd_indicator" in seg.columns else None

        # CCW prevalence (proportion of segment with each flag)
        for flag in present_ccw:
            row[f"pct_{flag}"] = round(100 * seg[flag].mean(), 1)

        # Utilization (prior year 2009)
        for col in ["ip_admissions_2009", "ip_days_2009", "ed_visits_2009",
                    "op_visits_2009", "carrier_visits_2009", "total_cost_2009"]:
            if col in seg.columns:
                row[f"mean_{col}"] = round(seg[col].mean(), 2)

        # Demographics
        row["mean_age"] = round(seg["age_at_2010"].mean(), 1) if "age_at_2010" in seg.columns else None
        row["pct_female"] = round(100 * (seg["sex"] == "2").mean(), 1) if "sex" in seg.columns else None
        row["pct_white"] = round(100 * (seg["race"] == "White").mean(), 1) if "race" in seg.columns else None
        row["pct_black"] = round(100 * (seg["race"] == "Black").mean(), 1) if "race" in seg.columns else None
        row["pct_hispanic"] = round(100 * (seg["race"] == "Hispanic").mean(), 1) if "race" in seg.columns else None
        row["mean_dual_eligible_months"] = round(seg["dual_eligible_months_2009"].mean(), 1) if "dual_eligible_months_2009" in seg.columns else None

        # Stability
        if not stability_df.empty:
            stab_row = stability_df[stability_df["cluster"] == c]
            row["mean_jaccard"] = stab_row["mean_jaccard"].values[0] if not stab_row.empty else None
            row["stability_tier"] = stab_row["stability_tier"].values[0] if not stab_row.empty else None

        rows.append(row)

    profile = pd.DataFrame(rows)

    # Add top-5 conditions per segment as a convenience column
    ccw_pct_cols = [f"pct_{f}" for f in present_ccw if f"pct_{f}" in profile.columns]
    if ccw_pct_cols:
        def top5(row):
            vals = {CCW_LABELS.get(c.replace("pct_", ""), c): row[c] for c in ccw_pct_cols}
            return ", ".join(f"{k} ({v:.0f}%)"
                             for k, v in sorted(vals.items(), key=lambda x: -x[1])[:5])
        profile["top5_conditions"] = profile.apply(top5, axis=1)

    log.info("Segment profile: %d segments", len(profile))
    return profile


def name_segments(profile: pd.DataFrame) -> pd.DataFrame:
    """Assign human-readable names based on each segment's signature *relative* to cohort mean.

    Absolute thresholds fail when cohort-level prevalences are high (e.g., CHF at 51%,
    Alzheimer's at 34%). Relative ratios identify what makes each segment distinctive.
    Decision order matters: rarer / more severe archetypes are checked first.
    """
    df = profile.copy()
    total_n = df["n"].sum()

    def wavg(col: str, fallback: float = 0.0) -> float:
        if col not in df.columns:
            return fallback
        return float((df[col] * df["n"]).sum() / total_n) if total_n > 0 else fallback

    mean_cost     = max(wavg("mean_total_cost_2010") or wavg("mean_total_cost_2009"), 1.0)
    mean_charlson = max(wavg("mean_charlson"), 0.01)
    mean_ip       = max(wavg("mean_ip_admissions_2009"), 0.01)
    mean_age      = wavg("mean_age", 75.0)

    ccw_pct_cols = [
        "pct_SP_CHF", "pct_SP_DIABETES", "pct_SP_COPD", "pct_SP_CNCR",
        "pct_SP_ALZHDMTA", "pct_SP_DEPRESSN", "pct_SP_ISCHMCHT",
        "pct_SP_CHRNKIDN", "pct_SP_STRKETIA", "pct_SP_RA_OA",
    ]
    ccw_means = {c: max(wavg(c), 1.0) for c in ccw_pct_cols if c in df.columns}

    names = []
    for _, row in df.iterrows():
        cost     = float(row.get("mean_total_cost_2010", row.get("mean_total_cost_2009", 0)) or 0)
        charlson = float(row.get("mean_charlson", 0) or 0)
        ip       = float(row.get("mean_ip_admissions_2009", 0) or 0)
        age      = float(row.get("mean_age", 75) or 75)

        def rel(col: str) -> float:
            return float(row.get(col, 0) or 0) / ccw_means.get(col, 1.0)

        rel_cost     = cost / mean_cost
        rel_charlson = charlson / mean_charlson
        rel_ip       = ip / mean_ip

        rel_cancer = rel("pct_SP_CNCR")
        rel_alz    = rel("pct_SP_ALZHDMTA")
        rel_chf    = rel("pct_SP_CHF")
        rel_copd   = rel("pct_SP_COPD")
        rel_dm     = rel("pct_SP_DIABETES")
        rel_dep    = rel("pct_SP_DEPRESSN")
        rel_ckd    = rel("pct_SP_CHRNKIDN")
        rel_stroke = rel("pct_SP_STRKETIA")

        # Advanced-Disease: Alzheimer's ≥ 1.6× cohort average (highly elevated, not just above avg)
        # or cancer ≥ 2× with elevated cost — checked first because it's the most serious archetype
        if rel_cost >= 1.1 and (rel_alz >= 1.6 or rel_cancer >= 2.0):
            name = "Advanced-Disease / End-of-Life"
        # High-Complexity, High-Utilization: high Charlson + high hospitalizations + elevated cost
        elif rel_charlson >= 1.3 and rel_ip >= 1.5 and rel_cost >= 1.1:
            name = "High-Complexity, High-Utilization"
        # High-Complexity, Low-Utilization: very sick (high Charlson) but almost no hospitalizations
        # — the "hidden burden" segment most actionable for proactive outreach
        elif rel_charlson >= 1.3 and rel_ip < 0.2:
            name = "High-Complexity, Low-Utilization"
        elif (rel_chf >= 1.3 or rel_copd >= 1.4) and rel_cost >= 1.0:
            name = "Cardiopulmonary-Dominant"
        elif rel_ckd >= 1.4 and rel_cost >= 1.0:
            name = "Renal / CKD-Dominant"
        elif rel_dm >= 1.25 and rel_dep >= 1.2:
            name = "Diabetes + Behavioral Health"
        elif rel_dm >= 1.2 and rel_cost < 0.9:
            name = "Rising-Risk Diabetic"
        elif age > mean_age + 3 and rel_cost < 0.75:
            name = "Elderly, Stable Low-Risk"
        # Two tiers of low-acuity: very low (rel_cost < 0.70) vs moderately low (< 0.90)
        elif rel_cost < 0.70 and rel_charlson < 0.70:
            name = "Low-Acuity, Preventive-Focus"
        elif rel_cost < 0.90 and rel_charlson < 0.70:
            name = "Well-Managed, Stable Chronic"
        elif rel_dep >= 1.35:
            name = "Depression-Dominant"
        elif rel_stroke >= 1.5:
            name = "Post-Stroke / Neurological"
        else:
            name = f"Segment {int(row['cluster']) + 1}"

        names.append(name)

    df["segment_name"] = names
    return df
