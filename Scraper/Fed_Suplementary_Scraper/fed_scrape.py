"""Fed counterpart to the ECB scrape.py -- same loop, manifest and layout.

Differences from the ECB version:
  * PDF support: press-conference transcripts and annual reports are PDFs,
    extracted with pdfplumber (falls back to pypdf).
        pip install pdfplumber        # or: pip install pypdf
  * get_text() targets the Fed's #article/#content region instead of the
    ECB's #main-content, and clean() trims Fed page furniture (usa-banner,
    mega-menu, 'Back to Top', 'Last Update:', footer). Like the ECB clean(),
    it is conservative -- eyeball a few files per source and tune the cut
    markers / junk lines as needed.

Usage:
    python fed_scrape.py                       # everything
    python fed_scrape.py fomc-statements feds-notes
"""

import os
import io
import re
import csv
import sys
import time

from bs4 import BeautifulSoup

from fed_tools import *  # pylint: disable=wildcard-import, unused-wildcard-import


def _pdf_text(content):
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n\n".join(filter(None, (p.extract_text() for p in pdf.pages)))
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(filter(None, (p.extract_text() for p in reader.pages)))
    except ImportError:
        raise RuntimeError("PDF source: pip install pdfplumber (or pypdf)")


def get_text(url):
    """Fetch a document and return its text. PDFs are extracted as-is;
    HTML pages return the body text of the main content region."""
    r = get(url)
    if url.lower().endswith(".pdf") or "pdf" in r.headers.get("Content-Type", ""):
        return _pdf_text(r.content), True
    soup = BeautifulSoup(r.text, "html.parser")
    main = soup.find(id="article") or soup.find(id="content") or soup.body or soup
    for tag in main.find_all(["script", "style", "nav", "header",
                              "footer", "aside", "form", "noscript"]):
        tag.decompose()
    # Fed-specific furniture that sits inside the content region
    for sel in (".usa-banner", ".header", ".t4-nav", ".breadcrumb",
                ".lastUpdate", ".skip", "#skipnav"):
        for tag in main.select(sel):
            tag.decompose()
    return re.sub(r"\n{3,}", "\n\n", main.get_text("\n")).strip(), False


# Everything from the first of these onward is footer/boilerplate.
_TAIL_CUTS = ("Back to Top", "Last Update:", "Stay Connected",
              "Board of Governors of the Federal Reserve System\n20th Street")
# Exact lines that are pure chrome wherever they appear.
_JUNK_LINES = {"Skip to main content",
               "An official website of the United States Government",
               "Here's how you know", "Back to Home",
               "Please enable JavaScript if it is disabled in your browser "
               "or access the information through the links provided below.",
               "Main Menu Toggle Button", "Sections Search Toggle Button",
               "Search", "Submit Search Button", "Search Submit Button Submit",
               "Toggle Dropdown Menu", "Advanced", "Watch Live", "Share",
               "RSS", "PDF", "HTML", "X", "Bluesky", "Home"}


def clean(text, is_pdf=False):
    """Trim Fed page furniture while PRESERVING paragraph breaks.
    PDFs get whitespace tidying only. Conservative -- tune as needed."""
    if not is_pdf:
        cut = min([text.find(m) for m in _TAIL_CUTS if m in text] + [len(text)])
        text = text[:cut]
        text = "\n".join(line for line in text.splitlines()
                         if line.strip() not in _JUNK_LINES)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scrape(source, overwrite=False):
    if source not in SOURCES:
        raise SystemExit(f"unknown source '{source}'; choose from: {', '.join(SOURCES)}")
    os.makedirs(CSV_DIR, exist_ok=True)

    items = get_index(source)
    print(f"[{source}] found {len(items)} documents")

    manifest = open(f"{CSV_DIR}/{source}_manifest.csv", "w", newline="")
    mw = csv.writer(manifest); mw.writerow(["date", "url", "clean_path", "status"])
    errorfile = open(f"{CSV_DIR}/{source}_missing.csv", "w", newline="")
    ew = csv.writer(errorfile); ew.writerow(["date", "url", "error"])

    for key, datestr, url in items:
        year = key[0]
        rpath = raw_path(source, year, datestr)
        cpath = clean_path(source, year, datestr)
        os.makedirs(os.path.dirname(rpath), exist_ok=True)
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        print(datestr, end=" ")
        if os.path.exists(cpath) and not overwrite:
            print("skip"); mw.writerow([datestr, url, cpath, "skipped"]); continue
        try:
            raw, is_pdf = get_text(url)
            if not raw.strip():
                raise ValueError("no text extracted")
            with open(rpath, "w") as f:
                f.write(raw)
            with open(cpath, "w") as f:
                f.write(clean(raw, is_pdf))
            print("y"); mw.writerow([datestr, url, cpath, "ok"])
        except Exception as e:  # pylint: disable=broad-except
            print("n"); ew.writerow([datestr, url, f"{type(e).__name__}: {e}"])
        time.sleep(1)

    manifest.close(); errorfile.close()


if __name__ == "__main__":
    targets = sys.argv[1:] or list(SOURCES)
    for src in targets:
        scrape(src)
