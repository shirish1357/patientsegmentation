"""Cohort definition: Medicare FFS beneficiaries with ≥1 chronic condition in 2009.

Inclusion criteria
------------------
1. Continuously enrolled in Parts A & B for all 12 months of 2009 AND 2010.
2. Not enrolled in HMO during 2009 or 2010 (claims would be incomplete).
3. At least one of the 11 CCW chronic-condition flags = 1 in the 2009 BSF.
4. Alive at 2010-01-01.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

CCW_FLAGS = [
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
    "SP_RA_OA", "SP_STRKETIA",
]


def build_cohort(bsf_2009: pd.DataFrame, bsf_2010: pd.DataFrame) -> pd.DataFrame:
    """Return a filtered BSF-2009 DataFrame for the analytic cohort.

    Parameters
    ----------
    bsf_2009 : BSF for 2009 (pre-loaded)
    bsf_2010 : BSF for 2010 (pre-loaded)

    Returns
    -------
    Filtered bsf_2009 with an added column ``in_cohort`` (all True in the result)
    and attrition counts printed to log.
    """
    df09 = bsf_2009.copy()
    df10 = bsf_2010[["DESYNPUF_ID", "BENE_HI_CVRAGE_TOT_MONS",
                      "BENE_SMI_CVRAGE_TOT_MONS", "BENE_HMO_CVRAGE_TOT_MONS",
                      "BENE_DEATH_DT"]].copy()

    n_start = len(df09)
    log.info("Cohort build: starting with %d beneficiaries in BSF 2009", n_start)

    # ── Filter 1: continuous Part A + Part B in 2009 ──────────────────────
    mask_ab_2009 = (df09["BENE_HI_CVRAGE_TOT_MONS"] == 12) & \
                   (df09["BENE_SMI_CVRAGE_TOT_MONS"] == 12)
    df09 = df09[mask_ab_2009]
    log.info("  After A/B 12-month 2009: %d (-%d)", len(df09), n_start - len(df09))

    # ── Filter 2: no HMO in 2009 ──────────────────────────────────────────
    if "BENE_HMO_CVRAGE_TOT_MONS" in df09.columns:
        df09 = df09[df09["BENE_HMO_CVRAGE_TOT_MONS"] == 0]
        log.info("  After no HMO 2009: %d", len(df09))

    # ── Merge 2010 enrollment to apply year-2010 filters ──────────────────
    df09 = df09.merge(df10, on="DESYNPUF_ID", how="inner", suffixes=("", "_2010"))
    log.info("  After inner join to BSF 2010: %d", len(df09))

    # ── Filter 3: continuous A/B in 2010 ──────────────────────────────────
    mask_ab_2010 = (df09["BENE_HI_CVRAGE_TOT_MONS_2010"] == 12) & \
                   (df09["BENE_SMI_CVRAGE_TOT_MONS_2010"] == 12)
    df09 = df09[mask_ab_2010]
    log.info("  After A/B 12-month 2010: %d", len(df09))

    # ── Filter 4: no HMO in 2010 ──────────────────────────────────────────
    if "BENE_HMO_CVRAGE_TOT_MONS_2010" in df09.columns:
        df09 = df09[df09["BENE_HMO_CVRAGE_TOT_MONS_2010"] == 0]
        log.info("  After no HMO 2010: %d", len(df09))

    # Drop the _2010 enrollment columns; keep only BENE_DEATH_DT_2010
    df09.drop(columns=[c for c in df09.columns
                       if c.endswith("_2010") and c != "BENE_DEATH_DT_2010"], inplace=True)

    # ── Filter 5: alive at 2010-01-01 ─────────────────────────────────────
    cutoff = pd.Timestamp("2010-01-01")
    df09["BENE_DEATH_DT"] = pd.to_datetime(df09["BENE_DEATH_DT"], format="%Y%m%d", errors="coerce")
    if "BENE_DEATH_DT_2010" in df09.columns:
        df09["BENE_DEATH_DT_2010"] = pd.to_datetime(df09["BENE_DEATH_DT_2010"], format="%Y%m%d", errors="coerce")

    death_dt = df09.get("BENE_DEATH_DT_2010", df09["BENE_DEATH_DT"])
    alive_mask = death_dt.isna() | (death_dt >= cutoff)
    df09 = df09[alive_mask]
    log.info("  After alive at 2010-01-01: %d", len(df09))

    # ── Filter 6: ≥1 chronic condition flag in 2009 ───────────────────────
    present_flags = [f for f in CCW_FLAGS if f in df09.columns]
    # CCW flags are coded 1=Yes, 2=No in SynPUF
    has_chronic = (df09[present_flags] == 1).any(axis=1)
    df09 = df09[has_chronic]
    log.info("  After ≥1 chronic condition: %d (final cohort)", len(df09))

    df09 = df09.reset_index(drop=True)
    return df09


def cohort_attrition_table(bsf_2009: pd.DataFrame, bsf_2010: pd.DataFrame) -> pd.DataFrame:
    """Return a step-by-step attrition table (for the briefing appendix)."""
    rows = []
    df09 = bsf_2009.copy()
    df10 = bsf_2010[["DESYNPUF_ID", "BENE_HI_CVRAGE_TOT_MONS",
                      "BENE_SMI_CVRAGE_TOT_MONS", "BENE_HMO_CVRAGE_TOT_MONS",
                      "BENE_DEATH_DT"]].copy()

    rows.append(("BSF 2009 — all beneficiaries", len(df09)))

    mask = (df09["BENE_HI_CVRAGE_TOT_MONS"] == 12) & (df09["BENE_SMI_CVRAGE_TOT_MONS"] == 12)
    df09 = df09[mask]
    rows.append(("Continuous A+B 2009 (12 months)", len(df09)))

    if "BENE_HMO_CVRAGE_TOT_MONS" in df09.columns:
        df09 = df09[df09["BENE_HMO_CVRAGE_TOT_MONS"] == 0]
    rows.append(("No HMO enrollment 2009", len(df09)))

    df09 = df09.merge(df10, on="DESYNPUF_ID", how="inner", suffixes=("", "_2010"))
    rows.append(("Also enrolled 2010 (inner join)", len(df09)))

    mask2 = (df09["BENE_HI_CVRAGE_TOT_MONS_2010"] == 12) & (df09["BENE_SMI_CVRAGE_TOT_MONS_2010"] == 12)
    df09 = df09[mask2]
    rows.append(("Continuous A+B 2010 (12 months)", len(df09)))

    if "BENE_HMO_CVRAGE_TOT_MONS_2010" in df09.columns:
        df09 = df09[df09["BENE_HMO_CVRAGE_TOT_MONS_2010"] == 0]
    rows.append(("No HMO enrollment 2010", len(df09)))

    cutoff = pd.Timestamp("2010-01-01")
    for col in ["BENE_DEATH_DT", "BENE_DEATH_DT_2010"]:
        if col in df09.columns:
            df09[col] = pd.to_datetime(df09[col], format="%Y%m%d", errors="coerce")
    death_dt = df09.get("BENE_DEATH_DT_2010", df09["BENE_DEATH_DT"])
    df09 = df09[death_dt.isna() | (death_dt >= cutoff)]
    rows.append(("Alive at 2010-01-01", len(df09)))

    present_flags = [f for f in CCW_FLAGS if f in df09.columns]
    df09 = df09[(df09[present_flags] == 1).any(axis=1)]
    rows.append(("≥1 CCW chronic condition in 2009", len(df09)))

    result = pd.DataFrame(rows, columns=["Step", "N"])
    result["Removed"] = result["N"].shift(1, fill_value=result["N"].iloc[0]) - result["N"]
    result.loc[0, "Removed"] = 0
    return result
