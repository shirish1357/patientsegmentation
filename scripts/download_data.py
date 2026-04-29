#!/usr/bin/env python3
"""Download CMS DE-SynPUF Sample 1 files (2008–2010) to data/raw/.

Usage
-----
    python scripts/download_data.py [--force]

Files downloaded
----------------
- Beneficiary Summary File (BSF): 3 years
- Inpatient Claims: 2008–2010 (multi-part)
- Outpatient Claims: 2008–2010
- Carrier Claims: 2008–2010 (multi-part)
- Prescription Drug Events (PDE): 2008–2010

Total download size: ~5–8 GB compressed. Allow 15–45 minutes on broadband.

Note: CMS occasionally changes the download portal. If URLs return 404, visit
      https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF
      and update the MANIFEST below.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

# ---------------------------------------------------------------------------
# Download manifest — file description, URL, expected zip name
# CMS hosts SynPUF on its website; URL format is stable but file names vary.
# ---------------------------------------------------------------------------
# URLs verified from CMS DE-SynPUF Sample 1 page (2026-04-29).
# BSF and IP/OP are served from www.cms.gov (lowercase paths required).
# Carrier and PDE are served from downloads.cms.gov/files/ (follow 301 redirects).
# NOTE: The 2010 BSF URL on the CMS page is a known CMS bug — it links to the
# sample_20 filename despite being labeled "Sample 1 2010 BSF". We download it
# as-is; if DESYNPUF_IDs don't match 2008/2009 cohort, outcomes fall back to claims.
CMS_BASE = "https://www.cms.gov/research-statistics-data-and-systems/downloadable-public-use-files/synpufs/downloads"
DL_BASE  = "http://downloads.cms.gov/files"

MANIFEST = [
    # Beneficiary Summary Files
    {"desc": "BSF 2008", "url": f"{CMS_BASE}/de1_0_2008_beneficiary_summary_file_sample_1.zip"},
    {"desc": "BSF 2009", "url": f"{CMS_BASE}/de1_0_2009_beneficiary_summary_file_sample_1.zip"},
    # CMS page bug: 2010 BSF for Sample 1 links to sample_20 filename — download anyway
    {"desc": "BSF 2010", "url": "https://www.cms.gov/research-statistics-data-and-systems/statistics-trends-and-reports/synpufs/downloads/de1_0_2010_beneficiary_summary_file_sample_20.zip"},
    # Inpatient Claims (3-year file)
    {"desc": "Inpatient 2008-2010", "url": f"{CMS_BASE}/de1_0_2008_to_2010_inpatient_claims_sample_1.zip"},
    # Outpatient Claims (3-year file)
    {"desc": "Outpatient 2008-2010", "url": f"{CMS_BASE}/de1_0_2008_to_2010_outpatient_claims_sample_1.zip"},
    # Carrier Claims (two parts, served from downloads.cms.gov)
    {"desc": "Carrier Part 1 (1A)", "url": f"{DL_BASE}/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.zip"},
    {"desc": "Carrier Part 2 (1B)", "url": f"{DL_BASE}/DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.zip"},
    # Prescription Drug Events
    {"desc": "PDE 2008-2010", "url": f"{DL_BASE}/DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.zip"},
]


def _download_file(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        log.info("Already exists — skipping: %s", dest.name)
        return dest
    log.info("Downloading: %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                          desc=dest.name, ncols=80) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def _unzip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            dest = out_dir / Path(member).name
            if dest.exists():
                log.info("Already extracted: %s", dest.name)
            else:
                zf.extract(member, out_dir)
                # Move to flat directory if nested
                nested = out_dir / member
                if nested != dest and nested.exists():
                    nested.rename(dest)
            extracted.append(dest)
    return extracted


def main(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    errors = []

    for entry in MANIFEST:
        url = entry["url"]
        zip_name = url.split("/")[-1]
        zip_path = RAW_DIR / zip_name

        try:
            _download_file(url, zip_path, force=force)
            _unzip(zip_path, RAW_DIR)
            log.info("✓ %s", entry["desc"])
        except requests.HTTPError as e:
            log.error("HTTP error for %s: %s", entry["desc"], e)
            errors.append(entry["desc"])
        except Exception as e:
            log.error("Failed %s: %s", entry["desc"], e)
            errors.append(entry["desc"])

    if errors:
        log.error("\nThe following files failed to download:")
        for e in errors:
            log.error("  - %s", e)
        log.error(
            "\nManual fallback: visit\n"
            "  https://www.cms.gov/Research-Statistics-Data-and-Systems/"
            "Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF\n"
            "and download Sample 1 files for 2008–2010 into data/raw/, then unzip."
        )
        sys.exit(1)
    else:
        log.info("\nAll files downloaded and extracted to %s", RAW_DIR.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    args = parser.parse_args()
    main(force=args.force)
