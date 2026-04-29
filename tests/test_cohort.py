"""Tests for cohort inclusion/exclusion logic."""

import pandas as pd

from patientseg.cohort import CCW_FLAGS, build_cohort


def _make_bsf(records: list[dict]) -> pd.DataFrame:
    defaults = {
        "BENE_HI_CVRAGE_TOT_MONS": 12,
        "BENE_SMI_CVRAGE_TOT_MONS": 12,
        "BENE_HMO_CVRAGE_TOT_MONS": 0,
        "BENE_DEATH_DT": None,
        "BENE_BIRTH_DT": "19300101",
        "BENE_SEX_IDENT_CD": 1,
        "BENE_RACE_CD": 1,
        "BENE_ESRD_IND": "N",
        "PLAN_CVRG_MOS_NUM": 0,
        **{f: 2 for f in CCW_FLAGS},  # all flags = 2 (No) by default
    }
    rows = [{**defaults, **r} for r in records]
    return pd.DataFrame(rows)


def test_basic_inclusion():
    """All-passing beneficiary should make it through."""
    bsf_09 = _make_bsf([{"DESYNPUF_ID": "A001", "SP_DIABETES": 1}])
    bsf_10 = _make_bsf([{"DESYNPUF_ID": "A001"}])
    cohort = build_cohort(bsf_09, bsf_10)
    assert len(cohort) == 1
    assert cohort["DESYNPUF_ID"].iloc[0] == "A001"


def test_hmo_exclusion_2009():
    """Beneficiary with HMO months in 2009 should be excluded."""
    bsf_09 = _make_bsf([
        {"DESYNPUF_ID": "A001", "SP_DIABETES": 1, "BENE_HMO_CVRAGE_TOT_MONS": 6},
        {"DESYNPUF_ID": "A002", "SP_DIABETES": 1},
    ])
    bsf_10 = _make_bsf([
        {"DESYNPUF_ID": "A001"},
        {"DESYNPUF_ID": "A002"},
    ])
    cohort = build_cohort(bsf_09, bsf_10)
    assert "A001" not in cohort["DESYNPUF_ID"].values
    assert "A002" in cohort["DESYNPUF_ID"].values


def test_no_chronic_condition_excluded():
    """Beneficiary with no chronic conditions should be excluded."""
    bsf_09 = _make_bsf([
        {"DESYNPUF_ID": "A001"},  # all CCW flags = 2 (No) by default
        {"DESYNPUF_ID": "A002", "SP_CHF": 1},
    ])
    bsf_10 = _make_bsf([
        {"DESYNPUF_ID": "A001"},
        {"DESYNPUF_ID": "A002"},
    ])
    cohort = build_cohort(bsf_09, bsf_10)
    assert "A001" not in cohort["DESYNPUF_ID"].values
    assert "A002" in cohort["DESYNPUF_ID"].values


def test_deceased_before_2010_excluded():
    """Beneficiary who died before 2010-01-01 should be excluded."""
    bsf_09 = _make_bsf([
        {"DESYNPUF_ID": "A001", "SP_DIABETES": 1},
        {"DESYNPUF_ID": "A002", "SP_DIABETES": 1},
    ])
    bsf_10 = _make_bsf([
        {"DESYNPUF_ID": "A001", "BENE_DEATH_DT": "20091201"},  # died Dec 2009
        {"DESYNPUF_ID": "A002", "BENE_DEATH_DT": None},
    ])
    cohort = build_cohort(bsf_09, bsf_10)
    assert "A001" not in cohort["DESYNPUF_ID"].values
    assert "A002" in cohort["DESYNPUF_ID"].values


def test_not_enrolled_in_2010_excluded():
    """Beneficiary not in BSF 2010 (inner join) should be excluded."""
    bsf_09 = _make_bsf([
        {"DESYNPUF_ID": "A001", "SP_DIABETES": 1},
        {"DESYNPUF_ID": "A002", "SP_DIABETES": 1},
    ])
    bsf_10 = _make_bsf([
        {"DESYNPUF_ID": "A002"},  # A001 absent from 2010
    ])
    cohort = build_cohort(bsf_09, bsf_10)
    assert "A001" not in cohort["DESYNPUF_ID"].values
    assert "A002" in cohort["DESYNPUF_ID"].values


def test_partial_ab_enrollment_excluded():
    """Beneficiary with fewer than 12 months of Part A in 2009 should be excluded."""
    bsf_09 = _make_bsf([
        {"DESYNPUF_ID": "A001", "SP_DIABETES": 1, "BENE_HI_CVRAGE_TOT_MONS": 6},
    ])
    bsf_10 = _make_bsf([{"DESYNPUF_ID": "A001"}])
    cohort = build_cohort(bsf_09, bsf_10)
    assert len(cohort) == 0
