"""PDF-based ECB corpus sources that don't fit the HTML-index pipeline.

Sources:
  monthly-bulletin   ECB Monthly Bulletin, Jan 1999 - Dec 2014 (Economic
                     Bulletin's predecessor; narrative text is extracted and
                     the statistical annex trimmed).       [slice: research]
  annual-report-archive  ECB Annual Report PDFs 1998-2014 (the HTML era from
                     ~2015 is handled by scrape.py source 'ecb-annual-report').
                                                           [slice: stability]
  mep-letters        ECB letters replying to MEP questions. [slice: deliberation]
  monetary-dialogue  Full Q&A transcripts of the quarterly ECON hearings of
                     the ECB President, hosted on europarl.europa.eu.
                                                           [slice: deliberation]

Requires: pip install requests beautifulsoup4 pymupdf
Usage:    python scrape_pdf.py [--list] [source ...]
"""
import os
import re
import csv
import sys
import time
import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from tools import ECB, CSV_DIR, get, head_ok, HEADERS

EUROPARL = "https://www.europarl.europa.eu"

# ---------------------------------------------------------------------------
# PDF download + text extraction

def fetch_pdf_text(url):
    """Download a PDF and return its extracted text (needs pymupdf)."""
    import fitz  # PyMuPDF
    r = get(url)
    doc = fitz.open(stream=r.content, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()

def tidy(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(l.strip() for l in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def write_doc(source, label, year, datestr, text, mw):
    cpath = f"clean/{source}/{year}/{datestr}-{label}.txt"
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    with open(cpath, "w") as f:
        f.write(text)
    mw.writerow([datestr, cpath, "ok"])
    return cpath

def open_manifests(source):
    os.makedirs(CSV_DIR, exist_ok=True)
    m = open(f"{CSV_DIR}/{source}_manifest.csv", "w", newline="")
    e = open(f"{CSV_DIR}/{source}_missing.csv", "w", newline="")
    mw, ew = csv.writer(m), csv.writer(e)
    mw.writerow(["date", "clean_path", "status"])
    ew.writerow(["date_or_url", "error"])
    return m, e, mw, ew

# ---------------------------------------------------------------------------
# 1. Monthly Bulletin, 1999-01 .. 2014-12. URL scheme verified:
#    https://www.ecb.europa.eu/pub/pdf/mobu/mb{YYYY}{MM}en.pdf

MB_URL = ECB + "/pub/pdf/mobu/mb{y}{m:02d}en.pdf"
# the narrative ends where the statistical annex begins; trim there
_MB_ANNEX = re.compile(r"EURO AREA STATISTICS|STATISTICS OF THE EURO AREA", re.I)

def iter_monthly_bulletin():
    for y in range(1999, 2015):
        for m in range(1, 13):
            yield f"{y}-{m:02d}", y, MB_URL.format(y=y, m=m)

def clean_mb(text):
    m = _MB_ANNEX.search(text, 3000)      # ignore table-of-contents mention
    if m and m.start() > len(text) * 0.2: # only trim if a real late section
        text = text[:m.start()]
    return tidy(text)

# ---------------------------------------------------------------------------
# 2. Annual Report PDFs, 1998-2014 (pre-hash naming; [best-effort]):
#    https://www.ecb.europa.eu/pub/pdf/annrep/ar{YYYY}en.pdf

AR_URL = ECB + "/pub/pdf/annrep/ar{y}en.pdf"

def iter_annual_report():
    for y in range(1998, 2015):
        yield str(y), y, AR_URL.format(y=y)

# ---------------------------------------------------------------------------
# 2b. Financial Stability Review PDF archive, Dec 2004 - Nov 2017 (May 2018
#    onward has a real HTML edition -- see 'financial-stability-review' in
#    tools.py). Confirmed legacy naming has NO 'ecb.' prefix:
#    https://www.ecb.europa.eu/pub/pdf/fsr/financialstabilityreview{YYYYMM}en.pdf
#    FSR was semi-annual throughout this window (June/Dec, later May/Nov).

FSR_URL = ECB + "/pub/pdf/fsr/financialstabilityreview{y}{m:02d}en.pdf"

def iter_fsr_archive():
    for y in range(2004, 2018):
        for m in (6, 12) if y < 2011 else (5, 11):
            if y == 2004 and m == 6:
                continue   # first edition was Dec 2004
            if y == 2017 and m == 5:
                pass       # both editions exist in 2017
            yield f"{y}-{m:02d}", y, FSR_URL.format(y=y, m=m)

# ---------------------------------------------------------------------------
# 3. Letters to MEPs: crawl the section index + per-year fragments for links
#    whose filename contains 'mepletter'; the date is embedded as YYMMDD.

MEP_INDEX = ECB + "/press/other-publications/mep-letters/html/index.en.html"
MEP_FRAGS = [ECB + "/press/other-publications/mep-letters/{year}/html/index_include.en.html",
             ECB + "/press/pubbydate/{year}/html/index_include.en.html"]
_MEP_RX = re.compile(r"mepletter(\d{6})[^\"']*?\.en\.pdf", re.I)

def iter_mep_letters():
    this_year = datetime.date.today().year
    found = {}
    pages = [MEP_INDEX] + [t.format(year=y) for t in MEP_FRAGS
                           for y in range(2014, this_year + 1)]
    for u in pages:
        try:
            soup = BeautifulSoup(get(u).text, "html.parser")
        except requests.RequestException:
            continue
        for a in soup.find_all("a", href=True):
            m = _MEP_RX.search(a["href"])
            if m:
                found[m.group(0)] = (m.group(1), urljoin(u, a["href"]))
    items = []
    for id6, url in found.values():
        yy, mm, dd = int(id6[:2]), int(id6[2:4]), int(id6[4:6])
        year = 2000 + yy if yy < 90 else 1900 + yy
        items.append((f"{year}-{mm:02d}-{dd:02d}", year, url))
    # same-day letters to different MEPs: disambiguate with -b, -c, ...
    items.sort()
    seen, out = {}, []
    for datestr, year, url in items:
        n = seen.get(datestr, 0); seen[datestr] = n + 1
        out.append((datestr + (f"-{chr(ord('a')+n)}" if n else ""), year, url))
    return out

# ---------------------------------------------------------------------------
# 4. Monetary Dialogue transcripts on europarl.europa.eu. Three page
#    generations; we crawl the current ECON page plus the term archives and
#    keep any English PDF link that looks like a Dialogue transcript. The
#    meeting date is read from the PDF itself (transcripts open with
#    'DD-MM-YYYY'), falling back to a date in the surrounding link text.

MD_PAGES = ([f"{EUROPARL}/committees/en/econ/econ-policies/monetary-dialogue"] +
            [f"{EUROPARL}/committees/en/archives/{t}/econ/econ-policies/monetary-dialogue"
             for t in (7, 8, 9)])
_MD_LINK = re.compile(r"transcript", re.I)
_MD_DATE = re.compile(r"(\d{1,2})[-./](\d{1,2})[-./](\d{4})")

def _md_links():
    found = {}
    for page in MD_PAGES:
        try:
            soup = BeautifulSoup(get(page).text, "html.parser")
        except requests.RequestException:
            print(f"  (could not read {page})")
            continue
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = " ".join(a.stripped_strings)
            if ".pdf" not in href.lower():
                continue
            if not (_MD_LINK.search(text) or _MD_LINK.search(href)):
                continue
            # keep the English version: EN in the label or _EN in the name
            blob = f"{text} {href}"
            if re.search(r"\bEN\b|_EN|%20EN", blob) or "DE" not in blob:
                found[href] = (urljoin(page, href), text)
    return list(found.values())

def scrape_monetary_dialogue(overwrite=False):
    source, label = "monetary-dialogue", "md"
    m, e, mw, ew = open_manifests(source)
    links = _md_links()
    print(f"[{source}] found {len(links)} candidate transcript PDFs")
    seen_dates = {}
    for url, text in links:
        try:
            body = fetch_pdf_text(url)
            dm = _MD_DATE.search(body[:3000]) or _MD_DATE.search(text)
            if not dm:
                raise ValueError("no date found in transcript")
            dd, mm, yyyy = (int(g) for g in dm.groups())
            if mm > 12:                      # tolerate MM-DD-YYYY orderings
                dd, mm = mm, dd
            datestr = f"{yyyy}-{mm:02d}-{dd:02d}"
            n = seen_dates.get(datestr, 0); seen_dates[datestr] = n + 1
            if n:
                datestr += f"-{chr(ord('a') + n)}"
            cpath = f"clean/{source}/{yyyy}/{datestr}-{label}.txt"
            if os.path.exists(cpath) and not overwrite:
                print(datestr, "skip"); mw.writerow([datestr, cpath, "skipped"]); continue
            write_doc(source, label, yyyy, datestr, tidy(body), mw)
            print(datestr, "y")
        except Exception as exc:  # pylint: disable=broad-except
            print("n", url)
            ew.writerow([url, f"{type(exc).__name__}: {exc}"])
        time.sleep(1)
    m.close(); e.close()

# ---------------------------------------------------------------------------

def scrape_constructed(source, label, items, cleaner=tidy, overwrite=False,
                       list_only=False):
    if list_only:
        avail = [(d, u) for d, _, u in items]
        print(f"[{source}] {len(avail)} constructed/discovered documents "
              f"{avail[0][0]} .. {avail[-1][0]}" if avail else f"[{source}] EMPTY")
        return
    m, e, mw, ew = open_manifests(source)
    print(f"[{source}] {len(items)} documents")
    for datestr, year, url in items:
        cpath = f"clean/{source}/{year}/{datestr}-{label}.txt"
        print(datestr, end=" ")
        if os.path.exists(cpath) and not overwrite:
            print("skip"); mw.writerow([datestr, cpath, "skipped"]); continue
        try:
            text = cleaner(fetch_pdf_text(url))
            if len(text) < 500:
                raise ValueError("suspiciously little text extracted")
            write_doc(source, label, year, datestr, text, mw)
            print("y")
        except Exception as exc:  # pylint: disable=broad-except
            print("n")
            ew.writerow([f"{datestr} {url}", f"{type(exc).__name__}: {exc}"])
        time.sleep(1)
    m.close(); e.close()

PDF_SOURCES = {
    "monthly-bulletin":      lambda lo: scrape_constructed(
        "monthly-bulletin", "mb", list(iter_monthly_bulletin()),
        cleaner=clean_mb, list_only=lo),
    "fsr-archive":           lambda lo: scrape_constructed(
        "fsr-archive", "fsr", list(iter_fsr_archive()), list_only=lo),
    "annual-report-archive": lambda lo: scrape_constructed(
        "annual-report-archive", "ar", list(iter_annual_report()), list_only=lo),
    "mep-letters":           lambda lo: scrape_constructed(
        "mep-letters", "mep", iter_mep_letters(), list_only=lo),
    "monetary-dialogue":     lambda lo: (
        print("[monetary-dialogue] discovery:") or
        print("\n".join(f"  {u}" for u, _ in _md_links())) if lo
        else scrape_monetary_dialogue()),
}

if __name__ == "__main__":
    args = sys.argv[1:]
    list_only = "--list" in args
    targets = [a for a in args if a != "--list"] or list(PDF_SOURCES)
    for src in targets:
        if src not in PDF_SOURCES:
            raise SystemExit(f"unknown source '{src}'; choose from: "
                             f"{', '.join(PDF_SOURCES)}")
        PDF_SOURCES[src](list_only)
