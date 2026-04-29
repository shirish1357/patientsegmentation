"""Map segment profiles to a care management prioritization framework.

Axes
----
Cost/Utilization: high (top third of cohort mean cost), medium, low
Modifiability: conditions with strong care-management evidence vs. less modifiable

Evidence base
-------------
- CMS Chronic Care Management (CCM) billing codes (99490, 99491, 99487)
- Transitional Care Management (TCM) codes (99495, 99496) for post-discharge
- Ambulatory Intensive Caring Unit (AICU) model for polychronic patients
- HEDIS quality measures for DM, CHF, COPD, depression
"""

from __future__ import annotations

import pandas as pd

# Conditions with strong care-management evidence (from published ACO / PCMH literature)
HIGH_MODIFIABILITY_CONDITIONS = {
    "SP_CHF", "SP_DIABETES", "SP_COPD", "SP_DEPRESSN", "SP_ISCHMCHT",
    "SP_CHRNKIDN", "SP_RA_OA",
}

# Conditions that are generally less modifiable via outreach
LOW_MODIFIABILITY_CONDITIONS = {
    "SP_CNCR", "SP_ALZHDMTA", "SP_STRKETIA",
}


def compute_modifiability_score(profile_row: pd.Series) -> float:
    """Return modifiability score (0–1) for a segment based on condition mix."""
    high_score = sum(
        profile_row.get(f"pct_{c}", 0) or 0
        for c in HIGH_MODIFIABILITY_CONDITIONS
    )
    low_score = sum(
        profile_row.get(f"pct_{c}", 0) or 0
        for c in LOW_MODIFIABILITY_CONDITIONS
    )
    total = high_score + low_score
    return high_score / total if total > 0 else 0.5


def assign_cost_tier(cost: float, cohort_mean: float) -> str:
    if cost > 1.15 * cohort_mean:
        return "High"
    elif cost > 0.75 * cohort_mean:
        return "Medium"
    else:
        return "Low"


OUTREACH_MATRIX = {
    # (cost_tier, modifiability_bucket) → (intensity, intervention_type, cms_codes)
    ("High", "High"): (
        "Intensive",
        "RN care manager + in-home assessment + Transitional Care Management (post-discharge)",
        "CPT 99487, 99489, 99495/99496",
    ),
    ("High", "Low"): (
        "Specialized",
        "Palliative care consult + social work + goals-of-care conversation",
        "CPT 99497/99498 (ACP), Hospice eligibility screening",
    ),
    ("Medium", "High"): (
        "Moderate",
        "Telephonic care management (monthly) + disease-specific self-management education",
        "CPT 99490/99491 (CCM), HEDIS DM/CHF measures",
    ),
    ("Medium", "Low"): (
        "Moderate",
        "Telephonic check-in (quarterly) + caregiver support resources",
        "CPT 99490 (CCM), caregiver assessment tool",
    ),
    ("Low", "High"): (
        "Light-Touch",
        "Annual wellness visit + preventive care reminders + patient portal outreach",
        "CPT G0438/G0439 (AWV), HEDIS preventive measures",
    ),
    ("Low", "Low"): (
        "Light-Touch",
        "Annual wellness visit + social determinants screening",
        "CPT G0438/G0439, ICD-10 Z-codes for SDOH",
    ),
}


def build_prioritization_table(profile: pd.DataFrame) -> pd.DataFrame:
    """Map each segment to an outreach intensity and intervention type.

    Parameters
    ----------
    profile : output of profile.name_segments()

    Returns
    -------
    DataFrame with prioritization columns appended.
    """
    df = profile.copy()

    cost_col = "mean_total_cost_2010"
    if cost_col not in df.columns:
        cost_col = "mean_total_cost_2009"

    cohort_mean_cost = df[cost_col].mean()

    prio_rows = []
    for _, row in df.iterrows():
        cost = row.get(cost_col, 0) or 0
        cost_tier = assign_cost_tier(cost, cohort_mean_cost)
        mod_score = compute_modifiability_score(row)
        mod_bucket = "High" if mod_score >= 0.5 else "Low"

        intensity, intervention, cms_codes = OUTREACH_MATRIX.get(
            (cost_tier, mod_bucket),
            ("Light-Touch", "Annual wellness visit", "CPT G0438/G0439"),
        )

        prio_rows.append({
            "mean_cost_2010": round(cost, 0),
            "cost_tier": cost_tier,
            "modifiability": mod_bucket,
            "outreach_intensity": intensity,
            "intervention_type": intervention,
            "cms_codes": cms_codes,
        })

    prio_df = pd.DataFrame(prio_rows, index=df.index)
    result = pd.concat([df, prio_df], axis=1)
    # Deduplicate columns that already exist in profile (keep profile value)
    result = result.loc[:, ~result.columns.duplicated(keep="first")]
    return result.sort_values("mean_cost_2010", ascending=False).reset_index(drop=True)
