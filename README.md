# Medicare Patient Segmentation — CMS DE-SynPUF

> **⚠ Synthetic data only.** This project uses CMS DE-SynPUF, which is fully synthetic and not clinically actionable. It is intended for portfolio and methodological demonstration only.

## Question

*Among Medicare beneficiaries with chronic conditions, which patient segments drive disproportionate cost and utilization — and how should a health system prioritize them for care management outreach?*

## Key Finding

*(Populated after running the pipeline)*

Among `N` Medicare fee-for-service beneficiaries with at least one chronic condition, the highest-cost segment — representing ~X% of the cohort — accounts for ~Y% of total 2010 Medicare spending. Four distinct patient segments emerge with meaningfully different clinical profiles, utilization patterns, and responses to care management, each requiring a different outreach strategy.

See [`briefing/exec_briefing.pdf`](briefing/exec_briefing.pdf) for the full 4–6 page executive briefing.

---

## Repository Structure

```
patientsegmentation/
├── data/
│   ├── raw/            # gitignored — populated by download script
│   └── processed/      # gitignored — populated by notebook
├── scripts/
│   └── download_data.py        # downloads DE-SynPUF Sample 1 (2008–2010)
├── src/patientseg/
│   ├── io.py           # CSV → Parquet conversion
│   ├── cohort.py       # cohort inclusion/exclusion
│   ├── features.py     # 3-dimension feature engineering
│   ├── charlson.py     # Quan ICD-9 Charlson Comorbidity Index
│   ├── cluster.py      # K-Means + GMM sweep, comparison
│   ├── stability.py    # Hennig-style Jaccard stability validation
│   ├── profile.py      # segment profiling tables
│   └── prioritize.py   # cost×modifiability → outreach framework
├── notebooks/
│   └── 01_pipeline.ipynb       # end-to-end reproducible analysis
├── briefing/
│   ├── exec_briefing.qmd       # Quarto source (4–6 page executive PDF)
│   ├── exec_briefing.pdf       # rendered briefing (committed)
│   └── figures/                # auto-generated from notebook
├── dashboard/
│   └── app.py          # Plotly Dash interactive dashboard
└── tests/
    ├── test_cohort.py
    ├── test_features.py
    └── test_stability.py
```

---

## Setup

### 1. Environment

Requires Python ≥ 3.11 and [Quarto CLI](https://quarto.org/docs/get-started/).

```bash
# Create virtual environment (using uv — fast)
pip install uv
uv venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install package + dependencies
uv pip install -e ".[dev]"

# Install Quarto (for the PDF briefing)
# macOS: brew install quarto
# Other: https://quarto.org/docs/get-started/
```

### 2. Download data

```bash
python scripts/download_data.py
```

Downloads DE-SynPUF Sample 1 (2008–2010) from CMS to `data/raw/`. **~5–8 GB compressed, allow 15–45 minutes.**

**Manual fallback:** If the automated download fails (CMS occasionally changes URLs), download files manually from:
> https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF

Download "Sample 1" files for Beneficiary Summary (2008–2010), Inpatient, Outpatient, Carrier, and PDE claims. Unzip into `data/raw/`.

---

## Run the Pipeline

```bash
# Option A: run the notebook (recommended — shows all outputs inline)
jupyter lab notebooks/01_pipeline.ipynb

# Option B: execute headlessly
jupyter nbconvert --to notebook --execute notebooks/01_pipeline.ipynb \
    --output notebooks/01_pipeline_executed.ipynb
```

This produces:
- `data/processed/cohort_features.parquet`
- `data/processed/segments.parquet` (used by briefing + dashboard)
- `data/processed/stability_results.parquet`
- `data/processed/model_comparison.parquet`
- `briefing/figures/` (PNG charts)

---

## Build the Executive Briefing (PDF)

```bash
quarto render briefing/exec_briefing.qmd
# Output: briefing/exec_briefing.pdf
```

---

## Launch the Dashboard

```bash
python dashboard/app.py
# Open http://localhost:8050
```

The dashboard shows segment-level cost, utilization, condition profiles, and the prioritization framework. A dropdown toggles between K-Means and GMM solutions.

---

## Run Tests

```bash
pytest tests/ -v
```

Tests cover cohort inclusion/exclusion logic, the Quan ICD-9 Charlson mapping, and the stability Jaccard calculation.

---

## Methods Summary

| Step | Detail |
|---|---|
| **Cohort** | FFS Medicare, continuous A+B enrollment 2009–2010, ≥1 CCW chronic condition, no HMO, alive 2010-01-01 |
| **Baseline window** | 2009 (features only) |
| **Outcome window** | 2010 (held out, used for profiling only) |
| **Feature dimensions** | Clinical complexity (Charlson, CCW flags, polypharmacy), Prior-year utilization (IP, ED, OP, carrier, cost), Demographics (age, sex, race, dual-eligibility) |
| **Algorithms** | K-Means (K=2–10) + GMM full-covariance (K=2–10) |
| **K selection** | Silhouette (K-Means), BIC (GMM) |
| **Stability** | Hennig-style cluster-wise Jaccard, 80% subsample × 20 iterations |
| **Algorithm choice** | Stability (Jaccard) → if tie, F-stat on 2010 cost |
| **Profiling** | Size, cost, utilization, top conditions, demographics per segment |
| **Prioritization** | Cost tier × modifiability → outreach intensity + intervention type + CMS billing codes |

---

## Limitations

- **Synthetic data:** DE-SynPUF is not real patient data; results are illustrative.
- **HMO exclusion:** FFS-only; not generalizable to managed-care populations.
- **Single sample:** Geographic variation estimates may be imprecise.
- **No SDOH:** Social determinants (housing, food, transportation) not captured in claims.
- **ICD-9 era:** 2008–2010; coding patterns differ from current ICD-10 practice.

---

## Data Source

CMS Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF)  
Free download, no data use agreement required.  
https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF
