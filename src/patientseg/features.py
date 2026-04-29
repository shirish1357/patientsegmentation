"""Feature engineering across three dimensions for the 2009 baseline window.

Dimensions
----------
1. Clinical complexity (CCW flags, Charlson, polypharmacy, ESRD)
2. Prior-year utilization (IP admits, IP days, ED visits, OP visits, carrier visits, cost)
3. Demographics (age, sex, race, dual-eligibility)

Outcome columns (2010) are appended for profiling but NEVER used in clustering.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .charlson import CCW_FLAGS as _CCW_FLAGS
from .charlson import compute_charlson

log = logging.getLogger(__name__)

CCW_FLAGS = _CCW_FLAGS

# Revenue center codes that identify ED visits in the OP file (CMS standard)
ED_REV_CODES = {str(c) for c in range(450, 460)} | {"981"}


def build_clinical_complexity(
    cohort: pd.DataFrame,
    ip_2009: pd.DataFrame,
    op_2009: pd.DataFrame,
    carrier_2009: pd.DataFrame,
    pde_2009: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return per-beneficiary clinical complexity features from the 2009 baseline."""
    bene_ids = cohort["DESYNPUF_ID"].unique()
    feat = pd.DataFrame({"DESYNPUF_ID": bene_ids})

    # ── CCW chronic-condition counts ───────────────────────────────────────
    present_flags = [f for f in CCW_FLAGS if f in cohort.columns]
    ccw = cohort[["DESYNPUF_ID"] + present_flags].copy()
    # CCW coded 1=Yes, 2=No; convert to binary 1/0
    for f in present_flags:
        ccw[f] = (ccw[f] == 1).astype(int)
    ccw["n_chronic_conditions"] = ccw[present_flags].sum(axis=1)
    feat = feat.merge(ccw[["DESYNPUF_ID", "n_chronic_conditions"] + present_flags],
                      on="DESYNPUF_ID", how="left")

    # ── ESRD indicator ─────────────────────────────────────────────────────
    if "BENE_ESRD_IND" in cohort.columns:
        esrd = cohort[["DESYNPUF_ID", "BENE_ESRD_IND"]].copy()
        esrd["esrd_indicator"] = (esrd["BENE_ESRD_IND"] == "Y").astype(int)
        feat = feat.merge(esrd[["DESYNPUF_ID", "esrd_indicator"]], on="DESYNPUF_ID", how="left")
    else:
        feat["esrd_indicator"] = 0

    # ── Charlson Comorbidity Index from IP + OP + Carrier dx codes ─────────
    ip_sub = ip_2009[ip_2009["DESYNPUF_ID"].isin(bene_ids)]
    op_sub = op_2009[op_2009["DESYNPUF_ID"].isin(bene_ids)]
    car_sub = carrier_2009[carrier_2009["DESYNPUF_ID"].isin(bene_ids)]

    dx_dfs = []
    for df in [ip_sub, op_sub, car_sub]:
        dx_cols = [c for c in df.columns if c.startswith("ICD9_DGNS_CD")]
        if dx_cols:
            dx_dfs.append(df[["DESYNPUF_ID"] + dx_cols])

    if dx_dfs:
        all_dx = pd.concat(dx_dfs, ignore_index=True)
        charlson_series = compute_charlson(all_dx)
        feat = feat.merge(charlson_series.reset_index(), on="DESYNPUF_ID", how="left")
        feat["charlson_index"] = feat["charlson_index"].fillna(0).astype(int)
    else:
        feat["charlson_index"] = 0

    # ── Polypharmacy (unique NDC proxy) from PDE ───────────────────────────
    if pde_2009 is not None and "PROD_SRVC_ID" in pde_2009.columns:
        pde_sub = pde_2009[pde_2009["DESYNPUF_ID"].isin(bene_ids)]
        poly = (
            pde_sub.groupby("DESYNPUF_ID")["PROD_SRVC_ID"]
            .nunique()
            .rename("n_unique_drugs_2009")
        )
        feat = feat.merge(poly.reset_index(), on="DESYNPUF_ID", how="left")
        feat["n_unique_drugs_2009"] = feat["n_unique_drugs_2009"].fillna(0).astype(int)
    else:
        feat["n_unique_drugs_2009"] = 0

    return feat


def build_utilization(
    cohort: pd.DataFrame,
    ip_2009: pd.DataFrame,
    op_2009: pd.DataFrame,
    carrier_2009: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-beneficiary prior-year (2009) utilization features."""
    bene_ids = cohort["DESYNPUF_ID"].unique()
    feat = pd.DataFrame({"DESYNPUF_ID": bene_ids})

    # ── Inpatient ──────────────────────────────────────────────────────────
    ip_sub = ip_2009[ip_2009["DESYNPUF_ID"].isin(bene_ids)]
    if not ip_sub.empty:
        ip_agg = ip_sub.groupby("DESYNPUF_ID").agg(
            ip_admissions_2009=("CLM_ID", "nunique"),
            ip_days_2009=("CLM_UTLZTN_DAY_CNT", "sum"),
        ).reset_index()
        feat = feat.merge(ip_agg, on="DESYNPUF_ID", how="left")
    feat["ip_admissions_2009"] = feat.get("ip_admissions_2009", pd.Series(0, index=feat.index)).fillna(0).astype(int)
    feat["ip_days_2009"] = feat.get("ip_days_2009", pd.Series(0, index=feat.index)).fillna(0).astype(int)

    # ── Outpatient (OP) + ED ───────────────────────────────────────────────
    op_sub = op_2009[op_2009["DESYNPUF_ID"].isin(bene_ids)].copy()
    if not op_sub.empty and "REV_CNTR" in op_sub.columns:
        op_sub["is_ed"] = op_sub["REV_CNTR"].astype(str).str.zfill(4).str[1:].isin(
            {str(c).zfill(3) for c in range(450, 460)} | {"981"}
        )
        ed_agg = (op_sub[op_sub["is_ed"]]
                  .groupby("DESYNPUF_ID")["CLM_ID"]
                  .nunique()
                  .rename("ed_visits_2009")
                  .reset_index())
        op_agg = (op_sub[~op_sub["is_ed"]]
                  .groupby("DESYNPUF_ID")["CLM_ID"]
                  .nunique()
                  .rename("op_visits_2009")
                  .reset_index())
        feat = feat.merge(ed_agg, on="DESYNPUF_ID", how="left")
        feat = feat.merge(op_agg, on="DESYNPUF_ID", how="left")
    else:
        # Fallback: no revenue code info, count all OP claims
        if not op_sub.empty:
            op_agg = (op_sub.groupby("DESYNPUF_ID")["CLM_ID"]
                      .nunique()
                      .rename("op_visits_2009")
                      .reset_index())
            feat = feat.merge(op_agg, on="DESYNPUF_ID", how="left")

    for col in ["ed_visits_2009", "op_visits_2009"]:
        if col not in feat.columns:
            feat[col] = 0
        feat[col] = feat[col].fillna(0).astype(int)

    # ── Carrier (physician / Part B) ───────────────────────────────────────
    car_sub = carrier_2009[carrier_2009["DESYNPUF_ID"].isin(bene_ids)]
    if not car_sub.empty:
        car_agg = (car_sub.groupby("DESYNPUF_ID")["CLM_ID"]
                   .nunique()
                   .rename("carrier_visits_2009")
                   .reset_index())
        feat = feat.merge(car_agg, on="DESYNPUF_ID", how="left")
    feat["carrier_visits_2009"] = feat.get("carrier_visits_2009", pd.Series(0)).fillna(0).astype(int)

    # ── Total cost from BSF roll-ups ───────────────────────────────────────
    cost_cols = [c for c in ["MEDREIMB_IP", "MEDREIMB_OP", "MEDREIMB_CAR"] if c in cohort.columns]
    if cost_cols:
        cost = cohort[["DESYNPUF_ID"] + cost_cols].copy()
        for c in cost_cols:
            cost[c] = pd.to_numeric(cost[c], errors="coerce").fillna(0)
        cost["total_cost_2009"] = cost[cost_cols].sum(axis=1)
        feat = feat.merge(cost[["DESYNPUF_ID", "total_cost_2009"]], on="DESYNPUF_ID", how="left")
    feat["total_cost_2009"] = feat.get("total_cost_2009", pd.Series(0.0)).fillna(0.0)

    return feat


def build_demographics(cohort: pd.DataFrame) -> pd.DataFrame:
    """Return per-beneficiary demographic features."""
    dem = cohort[["DESYNPUF_ID"]].copy()

    # Age as of 2010-01-01
    if "BENE_BIRTH_DT" in cohort.columns:
        birth = pd.to_datetime(cohort["BENE_BIRTH_DT"].astype(str), format="%Y%m%d", errors="coerce")
        dem["age_at_2010"] = ((pd.Timestamp("2010-01-01") - birth).dt.days / 365.25).round(1)
    else:
        dem["age_at_2010"] = np.nan

    if "BENE_SEX_IDENT_CD" in cohort.columns:
        dem["sex"] = cohort["BENE_SEX_IDENT_CD"].astype(str)
    else:
        dem["sex"] = "unknown"

    if "BENE_RACE_CD" in cohort.columns:
        race_map = {"1": "White", "2": "Black", "3": "Other", "4": "Asian",
                    "5": "Hispanic", "6": "North American Native"}
        dem["race"] = cohort["BENE_RACE_CD"].astype(str).map(race_map).fillna("Unknown")
    else:
        dem["race"] = "Unknown"

    # Dual eligibility: use PLAN_CVRG_MOS_NUM as proxy (months with Part D low-income subsidy)
    # Alternatively, beneficiary-level dual flag; using Part D LIS months as best available proxy
    if "PLAN_CVRG_MOS_NUM" in cohort.columns:
        dem["dual_eligible_months_2009"] = pd.to_numeric(
            cohort["PLAN_CVRG_MOS_NUM"], errors="coerce").fillna(0).astype(int)
    else:
        dem["dual_eligible_months_2009"] = 0

    return dem


def build_outcomes_2010(cohort_2010_bsf: pd.DataFrame, ip_2010: pd.DataFrame,
                         op_2010: pd.DataFrame) -> pd.DataFrame:
    """Build 2010 outcome columns (held out — never used in clustering)."""
    bene_ids = cohort_2010_bsf["DESYNPUF_ID"].unique()
    out = pd.DataFrame({"DESYNPUF_ID": bene_ids})

    # 2010 total cost from BSF
    cost_cols = [c for c in ["MEDREIMB_IP", "MEDREIMB_OP", "MEDREIMB_CAR"]
                 if c in cohort_2010_bsf.columns]
    if cost_cols:
        cost = cohort_2010_bsf[["DESYNPUF_ID"] + cost_cols].copy()
        for c in cost_cols:
            cost[c] = pd.to_numeric(cost[c], errors="coerce").fillna(0)
        cost["total_cost_2010"] = cost[cost_cols].sum(axis=1)
        out = out.merge(cost[["DESYNPUF_ID", "total_cost_2010"]], on="DESYNPUF_ID", how="left")
    out["total_cost_2010"] = out.get("total_cost_2010", pd.Series(0.0)).fillna(0.0)

    # IP admits 2010
    ip_sub = ip_2010[ip_2010["DESYNPUF_ID"].isin(bene_ids)]
    if not ip_sub.empty:
        ip_agg = (ip_sub.groupby("DESYNPUF_ID")["CLM_ID"]
                  .nunique()
                  .rename("ip_admissions_2010")
                  .reset_index())
        out = out.merge(ip_agg, on="DESYNPUF_ID", how="left")
    out["ip_admissions_2010"] = out.get("ip_admissions_2010", pd.Series(0)).fillna(0).astype(int)

    # ED visits 2010
    op_sub = op_2010[op_2010["DESYNPUF_ID"].isin(bene_ids)].copy()
    if not op_sub.empty and "REV_CNTR" in op_sub.columns:
        op_sub["is_ed"] = op_sub["REV_CNTR"].astype(str).str.zfill(4).str[1:].isin(
            {str(c).zfill(3) for c in range(450, 460)} | {"981"}
        )
        ed_agg = (op_sub[op_sub["is_ed"]]
                  .groupby("DESYNPUF_ID")["CLM_ID"]
                  .nunique()
                  .rename("ed_visits_2010")
                  .reset_index())
        out = out.merge(ed_agg, on="DESYNPUF_ID", how="left")
    out["ed_visits_2010"] = out.get("ed_visits_2010", pd.Series(0)).fillna(0).astype(int)

    # 30-day readmission flag: IP → next IP within 30 days for same bene
    if not ip_sub.empty and "CLM_FROM_DT" in ip_sub.columns and "CLM_THRU_DT" in ip_sub.columns:
        ip_dates = ip_sub[["DESYNPUF_ID", "CLM_FROM_DT", "CLM_THRU_DT"]].copy()
        for col in ["CLM_FROM_DT", "CLM_THRU_DT"]:
            ip_dates[col] = pd.to_datetime(ip_dates[col].astype(str), format="%Y%m%d", errors="coerce")
        ip_dates = ip_dates.sort_values(["DESYNPUF_ID", "CLM_THRU_DT"])
        ip_dates["prev_thru"] = ip_dates.groupby("DESYNPUF_ID")["CLM_THRU_DT"].shift(1)
        ip_dates["days_since_discharge"] = (ip_dates["CLM_FROM_DT"] - ip_dates["prev_thru"]).dt.days
        ip_dates["is_readmit"] = ip_dates["days_since_discharge"].between(1, 30)
        readmit = (ip_dates.groupby("DESYNPUF_ID")["is_readmit"]
                   .any()
                   .astype(int)
                   .rename("readmit_30d_2010")
                   .reset_index())
        out = out.merge(readmit, on="DESYNPUF_ID", how="left")
    out["readmit_30d_2010"] = out.get("readmit_30d_2010", pd.Series(0)).fillna(0).astype(int)

    return out


def assemble_feature_matrix(
    cohort: pd.DataFrame,
    ip_2009: pd.DataFrame,
    op_2009: pd.DataFrame,
    carrier_2009: pd.DataFrame,
    bsf_2010: pd.DataFrame,
    ip_2010: pd.DataFrame,
    op_2010: pd.DataFrame,
    pde_2009: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the complete feature matrix + outcome columns for the cohort."""
    log.info("Building clinical complexity features...")
    clinical = build_clinical_complexity(cohort, ip_2009, op_2009, carrier_2009, pde_2009)

    log.info("Building utilization features...")
    util = build_utilization(cohort, ip_2009, op_2009, carrier_2009)

    log.info("Building demographic features...")
    demo = build_demographics(cohort)

    log.info("Building 2010 outcome features (held out)...")
    outcomes = build_outcomes_2010(bsf_2010, ip_2010, op_2010)

    df = (clinical
          .merge(util, on="DESYNPUF_ID", how="inner")
          .merge(demo, on="DESYNPUF_ID", how="inner")
          .merge(outcomes, on="DESYNPUF_ID", how="left"))

    log.info("Feature matrix: %d rows × %d columns", *df.shape)
    return df
