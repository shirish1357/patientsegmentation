"""Quan ICD-9 mapping for the Charlson Comorbidity Index.

Reference: Quan H et al. (2005) Coding Algorithms for Defining Comorbidities
in ICD-9-CM and ICD-10 Administrative Data. Med Care 43(11):1130–1139.
"""

from __future__ import annotations

import pandas as pd

# CMS Chronic Conditions Warehouse flags present in the BSF
CCW_FLAGS = [
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
    "SP_RA_OA", "SP_STRKETIA",
]

# Condition name → (weight, list of ICD-9 prefix/regex patterns)
# Patterns are prefix strings unless they start with "^" (treated as regex).
_QUAN_ICD9: dict[str, tuple[int, list[str]]] = {
    "myocardial_infarction":        (1, ["410", "412"]),
    "congestive_heart_failure":     (1, ["39891", "4021", "4022", "4023", "4024", "4025",
                                         "4026", "4027", "4028", "4029", "425", "428",
                                         "4290", "4291", "4292", "4293", "4294", "4295",
                                         "4296", "4297", "4298", "4299"]),
    "peripheral_vascular_disease":  (1, ["0930", "4373", "440", "441", "4431", "4432",
                                         "4433", "4434", "4435", "4436", "4437", "4438",
                                         "4439", "4471", "557", "5571", "5579", "V434"]),
    "cerebrovascular_disease":      (1, ["36234", "430", "431", "432", "433", "434",
                                         "435", "436", "437", "438"]),
    "dementia":                     (1, ["290", "2941", "3312"]),
    "chronic_pulmonary_disease":    (1, ["4168", "4169", "490", "491", "492", "493",
                                         "494", "495", "496", "500", "501", "502",
                                         "503", "504", "505", "5064", "5081", "5088"]),
    "rheumatic_disease":            (1, ["4465", "7100", "7101", "7102", "7103", "7104",
                                         "7140", "7141", "7142", "7148", "725"]),
    "peptic_ulcer_disease":         (1, ["531", "532", "533", "534"]),
    "mild_liver_disease":           (1, ["07022", "07023", "07032", "07033", "07044",
                                         "07054", "0706", "0709", "570", "571", "5728",
                                         "5733", "5734", "5738", "5739", "V427"]),
    "diabetes_wo_complications":    (1, ["2500", "2501", "2502", "2503", "2508", "2509"]),
    "diabetes_w_complications":     (2, ["2504", "2505", "2506", "2507"]),
    "hemiplegia_paraplegia":        (2, ["3341", "342", "343", "3440", "3441", "3442",
                                         "3443", "3444", "3445", "3446", "3449"]),
    "renal_disease":                (2, ["40301", "40311", "40391", "40402", "40403",
                                         "40412", "40413", "40492", "40493", "582", "5830",
                                         "5831", "5832", "5834", "5836", "5837", "5838",
                                         "5839", "585", "586", "5880", "V420", "V451",
                                         "V560", "V561", "V562", "V568", "V45711", "V45719"]),
    "any_malignancy":               (2, ["140", "141", "142", "143", "144", "145",
                                         "146", "147", "148", "149", "150", "151", "152",
                                         "153", "154", "155", "156", "157", "158", "159",
                                         "160", "161", "162", "163", "164", "165", "166",
                                         "167", "168", "169", "170", "171", "172", "174",
                                         "175", "176", "177", "178", "179", "180", "181",
                                         "182", "183", "184", "185", "186", "187", "188",
                                         "189", "190", "191", "192", "193", "194", "195",
                                         "200", "201", "202", "203", "204", "205", "206",
                                         "207", "208", "2386"]),
    "moderate_severe_liver_disease": (3, ["4562", "4560", "4561", "5722", "5723", "5724",
                                          "5725", "5726", "5727", "5728"]),
    "metastatic_solid_tumor":        (6, ["196", "197", "198", "199"]),
    "aids_hiv":                      (6, ["042", "043", "044"]),
}


def _build_prefix_map() -> dict[str, tuple[str, int]]:
    """Build {prefix → (condition_name, weight)} for fast lookup."""
    out: dict[str, tuple[str, int]] = {}
    for condition, (weight, patterns) in _QUAN_ICD9.items():
        for p in patterns:
            out[p] = (condition, weight)
    return out


_PREFIX_MAP = _build_prefix_map()
_SORTED_PREFIXES = sorted(_PREFIX_MAP.keys(), key=len, reverse=True)


def code_to_conditions(icd9_code: str) -> list[str]:
    """Return list of Quan conditions matched by an ICD-9 code."""
    code = str(icd9_code).strip().replace(".", "")
    matched = []
    for prefix in _SORTED_PREFIXES:
        if code.startswith(prefix):
            matched.append(_PREFIX_MAP[prefix][0])
    return matched


def charlson_from_codes(codes: list[str]) -> int:
    """Compute Charlson score from a list of ICD-9 codes for one beneficiary.

    Handles the rule that diabetes_with_complications replaces
    diabetes_without_complications (they share a code family).
    """
    found: set[str] = set()
    for c in codes:
        found.update(code_to_conditions(c))

    # Hierarchy: diabetes_w_complications supersedes _wo_complications
    if "diabetes_w_complications" in found:
        found.discard("diabetes_wo_complications")
    # moderate_severe_liver supersedes mild_liver
    if "moderate_severe_liver_disease" in found:
        found.discard("mild_liver_disease")

    return sum(_QUAN_ICD9[c][0] for c in found)


def compute_charlson(dx_wide: pd.DataFrame, id_col: str = "DESYNPUF_ID") -> pd.Series:
    """Compute Charlson index for each beneficiary.

    Parameters
    ----------
    dx_wide : DataFrame with id_col + ICD9_DGNS_CD_1 … ICD9_DGNS_CD_N columns.
    id_col  : beneficiary ID column name.

    Returns
    -------
    Series indexed by id_col with integer Charlson scores.
    """
    dx_cols = [c for c in dx_wide.columns if c.startswith("ICD9_DGNS_CD")]

    # Stack all dx columns into a long series, drop nulls
    long = (
        dx_wide[[id_col] + dx_cols]
        .melt(id_vars=id_col, value_name="icd9")
        .dropna(subset=["icd9"])
    )
    long = long[long["icd9"].str.strip() != ""]

    # Map each code to conditions, then group by beneficiary
    long["conditions"] = long["icd9"].map(code_to_conditions)

    # Explode so each row is one (beneficiary, condition)
    long = long.explode("conditions").dropna(subset=["conditions"])

    if long.empty:
        return pd.Series(0, index=dx_wide[id_col].unique(), name="charlson_index")

    # Per beneficiary, collect unique conditions and compute score
    def _score(cond_set):
        cond_set = set(cond_set)
        if "diabetes_w_complications" in cond_set:
            cond_set.discard("diabetes_wo_complications")
        if "moderate_severe_liver_disease" in cond_set:
            cond_set.discard("mild_liver_disease")
        return sum(_QUAN_ICD9[c][0] for c in cond_set)

    scores = (
        long.groupby(id_col)["conditions"]
        .agg(lambda x: _score(set(x)))
        .rename("charlson_index")
    )
    # Beneficiaries with no matching codes get 0
    all_ids = dx_wide[id_col].unique()
    return scores.reindex(all_ids, fill_value=0)
