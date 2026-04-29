"""CSV → Parquet conversion and schema enforcement for DE-SynPUF files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = _PROJECT_ROOT / "data/raw"
PROCESSED_DIR = _PROJECT_ROOT / "data/processed"

# Columns needed from each file type — subset for memory efficiency
BSF_COLS = [
    "DESYNPUF_ID",
    "BENE_BIRTH_DT",
    "BENE_DEATH_DT",
    "BENE_SEX_IDENT_CD",
    "BENE_RACE_CD",
    "BENE_ESRD_IND",
    "SP_STATE_CODE",
    "BENE_COUNTY_CD",
    "BENE_HI_CVRAGE_TOT_MONS",
    "BENE_SMI_CVRAGE_TOT_MONS",
    "BENE_HMO_CVRAGE_TOT_MONS",
    "PLAN_CVRG_MOS_NUM",
    # CCW chronic condition flags
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
    "SP_RA_OA", "SP_STRKETIA",
    # Annual cost roll-ups (Medicare reimbursement only)
    "MEDREIMB_IP", "MEDREIMB_OP", "MEDREIMB_CAR",
    "BENRES_IP", "BENRES_OP", "BENRES_CAR",
    "PPPYMT_IP", "PPPYMT_OP", "PPPYMT_CAR",
]

IP_COLS = [
    "DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_THRU_DT",
    "CLM_PMT_AMT", "CLM_PASS_THRU_PER_DIEM_AMT", "NCH_BENE_IP_DDCTBL_AMT",
    "NCH_BENE_PTA_COINSRNC_LBLTY_AM",
    "NCH_IP_NCVRD_CHRG_AMT", "NCH_IP_TOT_DDCTN_AMT",
    "CLM_UTLZTN_DAY_CNT",
    "ICD9_DGNS_CD_1", "ICD9_DGNS_CD_2", "ICD9_DGNS_CD_3",
    "ICD9_DGNS_CD_4", "ICD9_DGNS_CD_5", "ICD9_DGNS_CD_6",
    "ICD9_DGNS_CD_7", "ICD9_DGNS_CD_8", "ICD9_DGNS_CD_9",
    "ICD9_DGNS_CD_10",
    "ICD9_PRCDR_CD_1", "ICD9_PRCDR_CD_2",
    "AT_PHYSN_NPI", "OP_PHYSN_NPI",
    "CLM_DRG_CD",
]

OP_COLS = [
    "DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_THRU_DT",
    "CLM_PMT_AMT", "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM",
    "ICD9_DGNS_CD_1", "ICD9_DGNS_CD_2", "ICD9_DGNS_CD_3",
    "ICD9_DGNS_CD_4", "ICD9_DGNS_CD_5", "ICD9_DGNS_CD_6",
    "ICD9_DGNS_CD_7", "ICD9_DGNS_CD_8", "ICD9_DGNS_CD_9",
    "ICD9_DGNS_CD_10",
    "REV_CNTR",
    "AT_PHYSN_NPI", "OP_PHYSN_NPI",
]

CARRIER_COLS = [
    "DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_THRU_DT",
    "ICD9_DGNS_CD_1", "ICD9_DGNS_CD_2",
    "LINE_NCH_PMT_AMT",
    "PRF_PHYSN_NPI_1",
    "HCPCS_CD_1",
]

PDE_COLS = [
    "DESYNPUF_ID", "PDE_ID", "SRVC_DT",
    "PROD_SRVC_ID",  # NDC code — used as drug proxy
    "QTY_DSPNSD_NUM", "DAYS_SUPLY_NUM",
    "PTNT_PAY_AMT", "TOT_RX_CST_AMT",
]

COLS_BY_TYPE = {
    "bsf": BSF_COLS,
    "ip": IP_COLS,
    "op": OP_COLS,
    "carrier": CARRIER_COLS,
    "pde": PDE_COLS,
}


_MULTI_YEAR_PATTERNS: dict[str, str] = {
    "ip": "*2008_to_2010*Inpatient*Sample_1*.csv",
    "op": "*2008_to_2010*Outpatient*Sample_1*.csv",
    "carrier": "*2008_to_2010*Carrier*Sample_1*.csv",
}


def _read_csv(path: Path, usecols: list[str]) -> pd.DataFrame:
    """Read a DE-SynPUF CSV, keeping only the requested columns that actually exist."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    keep = [c for c in usecols if c in header]
    missing = set(usecols) - set(keep)
    if missing:
        log.debug("Columns not in %s: %s", path.name, missing)
    return pd.read_csv(path, usecols=keep, low_memory=False)


def _filter_by_claim_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Keep only rows whose CLM_FROM_DT falls in `year` (date format YYYYMMDD)."""
    if "CLM_FROM_DT" not in df.columns:
        return df
    dt = pd.to_numeric(df["CLM_FROM_DT"], errors="coerce")
    return df[(dt // 10000) == year].copy()


def convert_to_parquet(file_type: str, year: int, force: bool = False) -> Path:
    """Read raw CSV for a given file type and year, write as Parquet.

    Returns the path to the written Parquet file.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{file_type}_{year}.parquet"
    if out_path.exists() and not force:
        log.info("Parquet already exists: %s", out_path)
        return out_path

    csv_files = sorted(RAW_DIR.glob(f"DE1_0_{year}_Beneficiary_Summary_File_Sample_1.csv")
                       if file_type == "bsf"
                       else RAW_DIR.glob(f"DE1_0_{year}*{file_type.upper()}*Sample_1*.csv"))

    if not csv_files:
        # Try lowercase / alternate naming
        patterns = {
            "bsf": f"*{year}*Beneficiary*Sample_1*.csv",
            "ip": f"*{year}*Inpatient*Sample_1*.csv",
            "op": f"*{year}*Outpatient*Sample_1*.csv",
            "carrier": f"*{year}*Carrier*Sample_1*.csv",
            "pde": f"*{year}*Prescription*Sample_1*.csv",
        }
        csv_files = sorted(RAW_DIR.glob(patterns[file_type]))

    if not csv_files and file_type in _MULTI_YEAR_PATTERNS:
        # SynPUF ships claim files as a single 2008-2010 span; filter by date below
        csv_files = sorted(RAW_DIR.glob(_MULTI_YEAR_PATTERNS[file_type]))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found for {file_type} {year} in {RAW_DIR}. "
            "Run scripts/download_data.py first."
        )

    usecols = COLS_BY_TYPE[file_type]
    frames = [_read_csv(f, usecols) for f in csv_files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # Claim files span multiple years — slice to the requested year
    if file_type in _MULTI_YEAR_PATTERNS:
        df = _filter_by_claim_year(df, year)
        if df.empty:
            raise FileNotFoundError(
                f"No {file_type} claims found for year {year} after date filtering."
            )

    df["year"] = year

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")
    log.info("Wrote %s rows → %s", len(df), out_path)
    return out_path


def load_parquet(file_type: str, year: int) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{file_type}_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run io.convert_to_parquet first.")
    return pd.read_parquet(path)
