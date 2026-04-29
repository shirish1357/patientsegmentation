# Medicare Patient Segmentation — CMS DE-SynPUF

> **Synthetic data only.** This project uses CMS DE-SynPUF, which is fully synthetic and not clinically actionable. It is a portfolio and methodological demonstration only.

[![CI](https://github.com/shirish1357/patientsegmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/shirish1357/patientsegmentation/actions/workflows/ci.yml)

---

## The Question

*Among Medicare fee-for-service beneficiaries with chronic conditions, which patient segments drive disproportionate cost and utilization — and how should a health system prioritize them for care management outreach?*

Patient segmentation is a foundational tool in population health management. Rather than treating all high-need patients the same way, it identifies meaningfully distinct subgroups — each with its own clinical profile, cost trajectory, and likely response to intervention. The goal here was to build and validate a segmentation pipeline rigorous enough to use in a real-world health system context, applied to a publicly available Medicare synthetic dataset.

---

## Key Results

**Cohort:** 45,305 Medicare fee-for-service beneficiaries with at least one chronic condition (2009 baseline, 2010 outcomes).

**Algorithm chosen:** K-Means, K=5, selected on stability grounds over GMM.

| Segment | N | Share | Mean 2010 Cost | Stability (Jaccard) |
|---|---|---|---|---|
| Low-Acuity Preventive-Focus | 10,778 | 23.8% | $2,415 | 0.81 — Stable |
| High-Complexity High-Utilization | 7,775 | 17.2% | $4,434 | 0.76 — Stable |
| High-Complexity Low-Utilization | 10,242 | 22.6% | $4,522 | 0.79 — Stable |
| Well-Managed Stable Chronic | 13,293 | 29.3% | $2,873 | 0.77 — Stable |
| Advanced-Disease / End-of-Life | 3,217 | 7.1% | $4,864 | 0.58 — Unstable |

The smallest segment (Advanced-Disease, 7%) carries the highest per-member cost and the lowest Jaccard stability — consistent with the clinical reality that end-of-life trajectories are highly variable and hard to replicate across subsamples. The two largest segments (Well-Managed Stable Chronic and Low-Acuity Preventive-Focus) represent 53% of the cohort at well-below-average cost, confirming that most chronic-disease beneficiaries are not high-need.

See `briefing/exec_briefing.qmd` for the full executive briefing (render with `quarto render briefing/exec_briefing.qmd`).

---

## Repository Structure

```
patientsegmentation/
├── .github/workflows/ci.yml   # GitHub Actions: lint + test on push/PR
├── src/patientseg/
│   ├── io.py                  # CSV → Parquet ingestion
│   ├── cohort.py              # cohort inclusion/exclusion with attrition table
│   ├── features.py            # 3-dimension feature engineering (2009 baseline)
│   ├── charlson.py            # Quan ICD-9 Charlson Comorbidity Index
│   ├── cluster.py             # K-Means + GMM sweep, K selection, comparison
│   ├── stability.py           # Hennig-style Jaccard stability validation
│   ├── profile.py             # segment profiling tables
│   └── prioritize.py          # cost × modifiability → outreach framework
├── notebooks/
│   └── 01_pipeline.ipynb      # end-to-end reproducible analysis
├── briefing/
│   └── exec_briefing.qmd      # Quarto source for 4–6 page executive PDF
├── dashboard/
│   └── app.py                 # Plotly Dash interactive dashboard
├── scripts/
│   └── download_data.py       # downloads DE-SynPUF Sample 1 from CMS
└── tests/
    ├── test_cohort.py          # cohort inclusion/exclusion logic
    ├── test_features.py        # Charlson mapping and CCI computation
    └── test_stability.py       # Jaccard similarity and stability scoring
```

---

## Methodology

### Data

**CMS DE-SynPUF (Data Entrepreneurs' Synthetic Public Use File), Sample 1, 2008–2010.**

This is a freely available, no-DUA-required synthetic Medicare dataset. It preserves realistic statistical distributions of claims — utilization patterns, condition prevalences, cost distributions — while containing no real patient data. It is the standard public dataset for Medicare methods development.

Files used:
- Beneficiary Summary File (BSF) 2009 and 2010
- Inpatient claims 2008–2010
- Outpatient claims 2008–2010
- Carrier (physician/supplier) claims 2008–2010

**Why 2009 as the baseline year and 2010 as the outcome year?**

Using features and outcomes from the same year would make cost predictions trivially circular — a beneficiary's utilization in a year is part of what drives cost in that year. By building features from 2009 and validating segment cost differences against 2010, we're asking whether the segments predict *future* cost, which is the operationally relevant question for care management targeting.

### Cohort Definition

Inclusion criteria applied in order:

1. **Continuous Part A + Part B enrollment in 2009** (12/12 months) — partial-year enrollment makes annualized cost comparisons invalid.
2. **No HMO enrollment in 2009** — HMO beneficiaries have incomplete fee-for-service claims; including them would severely undercount utilization.
3. **Inner join to 2010 BSF** — ensures the beneficiary is observable in the outcome year.
4. **Continuous Part A + Part B in 2010** (12/12 months) — same reason as criterion 1.
5. **No HMO enrollment in 2010** — same reason as criterion 2.
6. **Alive at 2010-01-01** — decedents cannot be managed prospectively; deaths during the baseline year are a separate population.
7. **At least one CCW chronic condition flag in 2009** — focuses the analysis on the population for which care management programs are designed.

The attrition table (logged during pipeline execution) documents how many beneficiaries are removed at each step.

### Feature Engineering

Features are engineered exclusively from the **2009 baseline window** and fall into three dimensions:

**1. Clinical complexity**
- 11 CCW chronic-condition binary flags (Alzheimer's, CHF, CKD, cancer, COPD, depression, diabetes, ischemic heart disease, osteoporosis, RA/OA, stroke/TIA) — coded 1/0 from BSF
- Count of chronic conditions (`n_chronic_conditions`)
- Charlson Comorbidity Index from ICD-9 diagnosis codes across inpatient, outpatient, and carrier claims, using the Quan (2005) mapping — this captures weighted severity, not just presence/absence
- Polypharmacy proxy: number of unique drug NDC codes from Part D claims
- ESRD indicator from BSF

**2. Prior-year utilization**
- Inpatient admissions and total inpatient days
- ED visits (identified via CMS revenue center codes 450–459 and 981 in outpatient claims)
- Outpatient visits
- Carrier (physician/supplier) visits
- Total Medicare-paid cost 2009

**3. Demographics**
- Age as of 2010-01-01
- Sex, race (from BSF)
- Dual-eligibility months 2009 (Medicare + Medicaid co-enrollment — a proxy for socioeconomic vulnerability)

**Why the Charlson index from claims, not just CCW flags?**

CCW flags are binary — a beneficiary either has diabetes or doesn't. The Charlson CCI uses diagnosis codes from actual claims to assign weighted severity scores across 17 condition categories. A beneficiary with diabetes *plus* renal complications scores higher than one with uncomplicated diabetes, even though both have the CCW diabetes flag set. This gives the clustering a richer signal of clinical burden.

**Why log1p-transform before scaling?**

Cost and utilization features are heavily right-skewed — most beneficiaries have zero or low values, with a long tail of very high utilizers. Without transformation, StandardScaler would compress most variation into the tail and give outliers disproportionate influence on cluster centroids. `log1p(x)` compresses the tail while preserving zero-values.

### Clustering

Both K-Means and GMM (full covariance) were swept across K=2–10 with fixed random seeds.

- **K-Means K selection:** highest silhouette score at K≥4. The floor of K=4 prevents the algorithm from returning trivially interpretable but clinically unhelpful solutions (K=2 almost always separates "high cost" from "low cost" — that is not actionable).
- **GMM K selection:** lowest BIC.

Final models were refit with more initializations (K-Means: n_init=50, GMM: n_init=10) to reduce sensitivity to initialization.

**Algorithm selection rule (stated before looking at results):**
1. If mean cluster-wise Jaccard stability differs by >0.05 between the two models, choose the stabler one.
2. Otherwise, choose the one with higher ANOVA F-statistic of 2010 costs across clusters (sharper cost discrimination = more useful for prioritization).

K-Means was chosen because it yielded higher mean Jaccard stability across all five clusters. GMM produced similar cluster structure (ARI was high) but with lower per-cluster reproducibility.

**Why not just use the highest silhouette score to pick the algorithm?**

Silhouette measures geometric compactness in the feature space. A model can have excellent silhouette but produce clusters that don't reproduce on subsamples — meaning it's overfitting to the specific random sample. Stability testing catches this. For a care management use case, reproducibility matters: segments that shift meaningfully with small data changes would make it hard to build reliable outreach programs around them.

### Stability Validation

Stability is assessed using Hennig (2007) cluster-wise Jaccard similarity:

1. Fit the chosen model on the full cohort → reference labels.
2. Repeat 20 times: sample 80% of the data (no replacement), refit, for each reference cluster find the subsample cluster with maximum Jaccard overlap.
3. Report mean Jaccard per reference cluster.

Jaccard ≥ 0.85: Highly stable. 0.60–0.84: Stable. <0.60: Unstable.

Four of five K-Means segments are Stable (Jaccard 0.76–0.81). The Advanced-Disease segment scores 0.58 (Unstable), which is expected — end-of-life trajectories are genuinely heterogeneous, so which specific patients land in this cluster varies with the subsample.

### Segment Prioritization

Each segment is mapped to a care management intervention using two axes:

- **Cost tier:** High (>1.15× cohort mean), Medium (0.75–1.15×), Low (<0.75×), based on mean 2010 cost
- **Modifiability:** proportion of the segment with conditions that have strong care-management evidence (CHF, diabetes, COPD, depression, CKD, IHD, RA/OA) vs. less-modifiable conditions (cancer, Alzheimer's, stroke/TIA)

The 6-cell matrix maps to outreach intensity (Intensive / Specialized / Moderate / Light-Touch) and intervention type, with relevant CMS billing codes (CCM 99490/99491/99487, TCM 99495/99496, ACP 99497/99498, AWV G0438/G0439).

---

## Setup

Requires Python ≥ 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Download data

```bash
python scripts/download_data.py
```

Downloads DE-SynPUF Sample 1 (2008–2010) from CMS to `data/raw/`. Allow 15–45 minutes. If the automated download fails (CMS occasionally changes URLs), download manually from:

> https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF

Download "Sample 1" files for Beneficiary Summary (2008–2010), Inpatient, Outpatient, and Carrier claims. Unzip into `data/raw/`.

---

## Run

**Pipeline (notebook):**
```bash
jupyter lab notebooks/01_pipeline.ipynb
```

Produces: `data/processed/cohort_features.parquet`, `data/processed/segments.parquet`, `data/processed/stability_results.parquet`, `data/processed/model_comparison.parquet`, and `briefing/figures/`.

**Executive briefing (PDF):**
```bash
quarto render briefing/exec_briefing.qmd
```

Requires [Quarto CLI](https://quarto.org/docs/get-started/) and a LaTeX distribution (TinyTeX: `quarto install tinytex`).

**Interactive dashboard:**
```bash
python dashboard/app.py
# Open http://localhost:8050
```

The dashboard displays segment-level cost, utilization, condition profiles, and the prioritization framework. A dropdown toggles between K-Means and GMM solutions.

**Tests:**
```bash
pytest tests/ -v
```

27 tests covering cohort inclusion/exclusion logic, the Quan ICD-9 Charlson mapping, and the Jaccard stability calculation. CI runs automatically on push and PR via GitHub Actions.

---

## Limitations

- **Synthetic data:** DE-SynPUF is not real patient data. Results are illustrative of the method, not clinical findings.
- **FFS only:** HMO/managed-care beneficiaries are excluded because their claims are incomplete in this dataset. Results do not generalize to that population.
- **Single geographic sample:** SynPUF Sample 1 is not a national probability sample; geographic variation estimates would be unreliable.
- **No social determinants:** Housing, food security, transportation, and other SDOH are not captured in claims.
- **ICD-9 era:** 2008–2010 data uses ICD-9 coding, which differs meaningfully from current ICD-10 practice. The Charlson mapping would need to be updated for post-2015 data.
- **Cluster 4 instability:** The Advanced-Disease segment (Jaccard 0.58) is operationally useful as a concept but its specific membership is not stable. Treat the segment archetype, not individual assignments, as the reliable output.

---

## Data Source

CMS Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF)
Free download, no data use agreement required.
https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF
