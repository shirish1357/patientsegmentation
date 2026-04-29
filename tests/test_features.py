"""Tests for Charlson mapping and feature engineering."""

import pandas as pd

from patientseg.charlson import charlson_from_codes, code_to_conditions, compute_charlson


class TestCharlsonMapping:
    def test_diabetes_without_complications(self):
        assert charlson_from_codes(["2500"]) == 1

    def test_diabetes_with_complications_supersedes(self):
        # DM with complications (weight 2) should supersede DM without (weight 1)
        score = charlson_from_codes(["2500", "2504"])
        assert score == 2

    def test_myocardial_infarction(self):
        assert charlson_from_codes(["410"]) == 1

    def test_metastatic_solid_tumor(self):
        assert charlson_from_codes(["196"]) == 6

    def test_aids_hiv(self):
        assert charlson_from_codes(["042"]) == 6

    def test_mild_liver_superseded_by_severe(self):
        # moderate_severe_liver (weight 3) supersedes mild_liver (weight 1)
        score = charlson_from_codes(["571", "5722"])  # mild + severe
        assert score == 3

    def test_multiple_conditions(self):
        # MI (1) + CHF (1) + diabetes_wo (1)
        score = charlson_from_codes(["410", "428", "2500"])
        assert score == 3

    def test_unknown_code_returns_zero(self):
        assert charlson_from_codes(["99999"]) == 0

    def test_empty_codes(self):
        assert charlson_from_codes([]) == 0

    def test_code_with_decimal(self):
        # Codes with decimal points should be handled
        conditions = code_to_conditions("250.0")
        assert "diabetes_wo_complications" in conditions


class TestComputeCharlson:
    def test_basic(self):
        dx_df = pd.DataFrame({
            "DESYNPUF_ID": ["B001", "B001", "B002"],
            "ICD9_DGNS_CD_1": ["410", "428", "2500"],
            "ICD9_DGNS_CD_2": [None, None, None],
        })
        scores = compute_charlson(dx_df)
        # B001: MI (1) + CHF (1) = 2
        assert scores["B001"] == 2
        # B002: diabetes_wo (1) = 1
        assert scores["B002"] == 1

    def test_missing_beneficiary_gets_zero(self):
        dx_df = pd.DataFrame({
            "DESYNPUF_ID": ["B001"],
            "ICD9_DGNS_CD_1": ["410"],
        })
        # Also check a bene with no claims
        scores = compute_charlson(pd.concat([
            dx_df,
            pd.DataFrame({"DESYNPUF_ID": ["B002"], "ICD9_DGNS_CD_1": [None]}),
        ], ignore_index=True))
        assert scores.get("B002", 0) == 0
