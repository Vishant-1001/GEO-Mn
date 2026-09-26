"""Acquire MOIL Ltd. public "Quantitative Details" disclosures and build a real production series.

Source: MOIL Ltd. investor relations, "Quantitative Details" (non-statutory disclosure),
  list API  POST https://backend.moil.nic.in/investor-relation/get-financial-public-list
            {"page": n, "lang": "english", "type": "quantitative-details", "searchString": ""}
  files     https://backend.moil.nic.in/getFiles/<pdf path>
Provenance: REAL_MOIL_PUBLIC (company-level; figures stated by MOIL as rounded and partly un-audited).

Each document reports manganese-ore PRODUCTION (non-fines, fines, grand total) for the reporting
period and the comparable period of the previous year, and usually the previous full FY.
Reporting periods are cumulative (April-to-date) except a few older "quarter ended" /
"half year ended" layouts, which are handled explicitly. Quarterly values are DERIVED by
differencing cumulative values within a fiscal year; where two documents state the same
cumulative figure the values are cross-checked.

Outputs
  data/raw/production/moil/*.pdf                         original documents (small, committed)
  data/processed/production/moil_production_disclosures.csv   one row per stated figure
  data/processed/production/moil_quarterly_production.csv     derived FY-quarter series
  data/manifests/real_moil_production.json

Run: python scripts/data_acquisition/fetch_moil_production.py [--offline]
(--offline re-parses the committed PDFs without network access.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "production" / "moil"
PROC = ROOT / "data" / "processed" / "production"
LIST_API = "https://backend.moil.nic.in/investor-relation/get-financial-public-list"
FILE_BASE = "https://backend.moil.nic.in/getFiles/"
LAKH = 100_000.0


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                                                                 "User-Agent": "GEO-MN research prototype"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def fetch_list():
    items, page = [], 1
    while True:
        d = _post(LIST_API, {"page": page, "lang": "english", "type": "quantitative-details", "searchString": ""})
        items += d.get("result") or []
        if page >= int(d.get("totalPages") or 1):
            return items
        page += 1


def slug(title):
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:90] + ".pdf"


def download(items):
    RAW.mkdir(parents=True, exist_ok=True)
    index = []
    for it in items:
        name = slug(it["title"])
        dest = RAW / name
        if not dest.exists():
            url = FILE_BASE + urllib.parse.quote(it["pdf"])
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "GEO-MN"}), timeout=120) as r:
                dest.write_bytes(r.read())
        index.append({"title": it["title"], "file": name, "source_url": FILE_BASE + it["pdf"]})
    (RAW / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def pdf_text(path):
    import pdfplumber

    with pdfplumber.open(path) as p:
        return "\n".join((pg.extract_text() or "") for pg in p.pages)


def _dmy(s):
    d, m, y = (int(x) for x in re.split(r"[./]", s))
    return date(y, m, d)


def fy_of(d: date) -> int:
    """Fiscal year label = calendar year in which the FY starts (FY 2024-25 -> 2024)."""
    return d.year if d.month >= 4 else d.year - 1


def months_into_fy(d: date) -> int:
    return (d.month - 4) % 12 + 1


def _row(text, label):
    """Numbers following a row label in the Production section only."""
    prod = text.split("Sales")[0]
    m = re.search(label + r"[^\n]*", prod)
    if not m:
        return None
    nums = re.findall(r"\d[\d,]*\.?\d*", m.group(0))
    return [float(n.replace(",", "")) for n in nums]


def parse_document(text, title):
    """Return the stated figures with explicit period semantics.

    Semantics come from the production table's COLUMN HEADER, not the document title
    (several "quarter ended" documents actually report six/nine-month cumulative values).
    Units come from magnitude: values < 100 are lakh tonnes even when mislabelled "MT".
    """
    prod = text.split("Sales")[0]
    header = " ".join(prod.split("Non-fines")[0].split("Production", 1)[-1].split())
    title_head = " ".join(text.split("Production")[0].split())
    gt, nf, fn = _row(text, r"Grand Total"), _row(text, r"Non-fines total"), _row(text, r"Fines ")
    if not gt:
        return []
    scale = LAKH if max(gt) < 100 else 1.0
    dates = [_dmy(d) for d in re.findall(r"\d{1,2}[./]\d{1,2}[./]\d{4}", header)]
    out = []

    def add(kind, start, end, idx, note):
        if idx >= len(gt):
            return
        out.append({
            "document": title, "period_kind": kind, "period_start": start.isoformat(), "period_end": end.isoformat(),
            "fy": fy_of(end), "months": (end.year - start.year) * 12 + end.month - start.month + 1,
            "grand_total_t": round(gt[idx] * scale, 1),
            "non_fines_t": round(nf[idx] * scale, 1) if nf and idx < len(nf) else None,
            "fines_t": round(fn[idx] * scale, 1) if fn and idx < len(fn) else None,
            "stated_unit": "lakh tonnes" if scale == LAKH else "tonnes", "column_header": header[:160], "note": note,
        })

    def fy_start(d):
        return date(fy_of(d), 4, 1)

    h = header.lower()
    # Only an explicit "quarter ended" column header means quarterly values. Bare date columns,
    # "six/nine months ended", "half year" and "period" documents are April-to-date (verified
    # against magnitudes: e.g. 30.09.2017 = 5.15 lakh t after Q1 = 2.68 lakh t).
    cumulative = "quarter ended" not in h
    if dates and dates[0].month != 3 and not ("financial year" in h and "quarter" not in h and not cumulative and len(dates) < 2):
        end = dates[0]
        prev_end = date(end.year - 1, end.month, end.day)
        if cumulative or end.month == 6:     # Q1 quarter == April-to-date
            add("CUMULATIVE", fy_start(end), end, 0, f"stated April-to-date ({header[:40]})")
            add("CUMULATIVE", fy_start(prev_end), prev_end, 1, "stated April-to-date (previous year)")
        else:
            qs = date(end.year, end.month - 2, 1)
            add("QUARTER", qs, end, 0, "stated quarter")
            add("QUARTER", date(qs.year - 1, qs.month, 1), prev_end, 1, "stated quarter (previous year)")
        add("FY", date(fy_of(end) - 1, 4, 1), date(fy_of(end), 3, 31), 2, "stated previous FY")
        return out
    # annual documents
    m = re.search(r"(\d{4})-(\d{2})", title_head) or re.search(r"(\d{4})-(\d{2})", header)
    end = dates[0] if dates and dates[0].month == 3 else (date(int(m.group(1)) + 1, 3, 31) if m else None)
    if end is None:
        return []
    if "quarter" in h and len(gt) >= 4:   # quarter-ended + FY layout (year ended 31.03.2014)
        add("QUARTER", date(end.year, 1, 1), end, 0, "stated quarter (Jan-Mar)")
        add("QUARTER", date(end.year - 1, 1, 1), date(end.year - 1, 3, 31), 1, "stated quarter (previous year)")
        add("FY", date(end.year - 1, 4, 1), end, 2, "stated FY")
        add("FY", date(end.year - 2, 4, 1), date(end.year - 1, 3, 31), 3, "stated previous FY")
    else:
        add("FY", date(end.year - 1, 4, 1), end, 0, "stated FY")
        add("FY", date(end.year - 2, 4, 1), date(end.year - 1, 3, 31), 1, "stated previous FY")
    return out


def build_quarterly(disc: pd.DataFrame):
    """Cumulative-to-quarter conversion with cross-checks. Returns (quarterly df, checks)."""
    cum = {}
    checks = []
    for r in disc.itertuples():
        end = pd.Timestamp(r.period_end)
        if r.period_kind in ("CUMULATIVE", "FY"):
            key = (r.fy, months_into_fy(end.date()))
            if key in cum and abs(cum[key]["t"] - r.grand_total_t) > 0.011 * LAKH:
                checks.append({"fy": r.fy, "months": key[1], "a": cum[key]["t"], "b": r.grand_total_t,
                               "docs": [cum[key]["doc"], r.document]})
            cum.setdefault(key, {"t": r.grand_total_t, "nf": r.non_fines_t, "fn": r.fines_t, "doc": f"{r.document} [{r.note}]"})
    quarters = {}
    for r in disc[disc["period_kind"] == "QUARTER"].itertuples():
        end = pd.Timestamp(r.period_end).date()
        quarters[(r.fy, months_into_fy(end) // 3)] = {"t": r.grand_total_t, "nf": r.non_fines_t, "fn": r.fines_t,
                                                     "basis": "stated quarter", "doc": f"{r.document} [{r.note}]"}
    for (fy, m), v in cum.items():
        if m % 3:
            continue
        q = m // 3
        prev = cum.get((fy, m - 3)) if m > 3 else {"t": 0.0, "nf": 0.0, "fn": 0.0, "doc": "-"}
        if prev is None or (fy, q) in quarters:
            continue
        sub = lambda a, b: None if a is None or b is None else round(a - b, 1)  # noqa: E731
        quarters[(fy, q)] = {"t": sub(v["t"], prev["t"]), "nf": sub(v["nf"], prev["nf"]), "fn": sub(v["fn"], prev["fn"]),
                             "basis": "stated cumulative" if q == 1 else "derived: cumulative difference",
                             "doc": v["doc"] + ("" if q == 1 else f" minus {prev['doc']}")}
    rows = []
    for (fy, q), v in sorted(quarters.items()):
        start_month = 4 + 3 * (q - 1)
        y = fy if start_month <= 12 else fy + 1
        sm = (start_month - 1) % 12 + 1
        start = date(y if sm >= 4 else fy + 1, sm, 1)
        rows.append({"fy": f"{fy}-{str(fy + 1)[-2:]}", "fy_start_year": fy, "fy_quarter": q,
                     "quarter_start": start.isoformat(), "production_t": v["t"], "non_fines_t": v["nf"], "fines_t": v["fn"],
                     "basis": v["basis"], "source_documents": v["doc"], "producer": "MOIL Ltd.",
                     "scope": "company total (all mines)", "mineral": "manganese ore", "temporal_resolution": "quarter",
                     "provenance": "REAL_MOIL_PUBLIC"})
    q = pd.DataFrame(rows).sort_values(["fy_start_year", "fy_quarter"]).reset_index(drop=True)
    # sanity: a quarter must be positive and below 40 % of its fiscal year's stated total
    fy_tot = {k[0]: v["t"] for k, v in cum.items() if k[1] == 12}
    for r in q.itertuples():
        tot = fy_tot.get(r.fy_start_year)
        if r.production_t is None or r.production_t <= 0 or (tot and r.production_t > 0.4 * tot):
            checks.append({"fy": r.fy, "quarter": r.fy_quarter, "problem": "implausible derived quarter",
                           "value": r.production_t, "fy_total": tot})
    return q, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        index = json.loads((RAW / "index.json").read_text())
    else:
        index = download(fetch_list())
    rows = []
    for it in index:
        rows += [dict(r, source_url=it["source_url"], file=it["file"]) for r in parse_document(pdf_text(RAW / it["file"]), it["title"])]
    disc = pd.DataFrame(rows).sort_values(["period_end", "period_kind"]).reset_index(drop=True)
    PROC.mkdir(parents=True, exist_ok=True)
    disc["provenance"] = "REAL_MOIL_PUBLIC"
    disc.to_csv(PROC / "moil_production_disclosures.csv", index=False)
    q, checks = build_quarterly(disc)
    q.to_csv(PROC / "moil_quarterly_production.csv", index=False)
    man = {
        "dataset_name": "MOIL Ltd. quantitative details — manganese ore production",
        "provider": "MOIL Ltd. (Government of India undertaking, Ministry of Steel)",
        "official_url": "https://moil.nic.in/public/investor-relations/financials",
        "retrieval_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": "public investor disclosure (non-statutory; figures rounded, partly un-audited per MOIL note)",
        "documents": len(index),
        "stated_figures": int(len(disc)),
        "quarters_derived": int(len(q)),
        "coverage": [q["quarter_start"].min(), q["quarter_start"].max()],
        "cross_check_conflicts": checks,
        "native_resolution": "company-level; cumulative April-to-date periods (quarterly derived by differencing)",
        "units": "tonnes (converted from lakh tonnes x 100,000 or MT)",
        "provenance": "REAL_MOIL_PUBLIC",
        "sha256": {it["file"]: hashlib.sha256((RAW / it["file"]).read_bytes()).hexdigest() for it in index},
    }
    (ROOT / "data" / "manifests" / "real_moil_production.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"{len(index)} documents, {len(disc)} stated figures, {len(q)} quarters, {len(checks)} cross-check conflicts")
    print(q[["fy", "fy_quarter", "production_t", "basis"]].to_string())


if __name__ == "__main__":
    sys.exit(main())
