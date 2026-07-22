import os
import re
import csv
import sys
import time

from bs4 import BeautifulSoup

from tools import * # pylint: disable=wildcard-import, unused-wildcard-import

def get_text(url):
    """Fetch a page and return the body text from its main content region."""
    soup = BeautifulSoup(get(url).text, "html.parser")
    main = soup.find(id="main-content") or soup.body or soup
    for tag in main.find_all(["script", "style", "nav", "header",
                              "footer", "aside", "form", "noscript"]):
        tag.decompose()
    return re.sub(r"\n{3,}", "\n\n", main.get_text("\n")).strip()

# ---------------------------------------------------------------------------
# Cleaning. ECB pages sandwich the document between (a) a search/nav header
# ending with the menu items Home ... Careers and (b) a footer starting at
# CONTACT / feedback / cookie banners. We locate the real start (the document
# title or, failing that, the greeting line) and the first footer marker, and
# keep only what lies between. Everything is matched on whole trimmed lines,
# so body paragraphs that merely mention these words are never touched.

_TITLE_RX = re.compile(
    r"^(?:introductory statement|monetary policy statement|press conference\b"
    r"|account of the monetary policy|meeting of \d|economic bulletin"
    r"|financial stability review|macroeconomic projections"
    r"|eurosystem staff|ecb staff)", re.I)
_GREETING_RX = re.compile(r"^(?:good afternoon|ladies and gentlemen)", re.I)

_END_EXACT = {"CONTACT", "Media contacts", "Are you happy with this page?",
              "Our website uses cookies", "All pages in this section",
              "SEE ALSO", "Related topics", "Disclaimer"}
_END_PREFIX = ("Reproduction is permitted", "Thank you for letting us know",
               "We use functional cookies", "We are always working to improve",
               "Copyright 19", "Copyright 20")

_DROP_EXACT = {"Jump to the transcript of the questions and answers"}

def clean(text):
    """Keep only the document body (statement + Q&A for press conferences),
    dropping ECB page furniture. Whitespace is tidied, paragraph breaks kept."""
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines()]

    # (a) where does the document start?
    start = None
    for rx in (_TITLE_RX, _GREETING_RX):
        start = next((i for i, l in enumerate(lines) if rx.match(l)), None)
        if start is not None:
            break
    if start is None:                       # fallback: cut the nav menu block
        start = 0
        for i, l in enumerate(lines[:150]):
            if l == "Careers":
                start = i + 1
                break

    # (b) where does the footer start?
    end = len(lines)
    for i in range(start + 1, len(lines)):
        l = lines[i]
        if l in _END_EXACT or l.startswith(_END_PREFIX):
            end = i
            break

    body = [l for l in lines[start:end] if l not in _DROP_EXACT]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(body)).strip()

# Paper slices (Table 3) -> scraper sources. PDF-era sources (Monthly
# Bulletin 1999-2014, Annual Report 1998-2014, MEP letters, Monetary
# Dialogue transcripts) live in scrape_pdf.py -- run both for full coverage.
GROUPS = {
    "policy-decisions":     ["monetary-policy-decisions", "other-gc-decisions"],
    "deliberation-proxy":   ["press-conference-statements", "monetary-policy-accounts"],
    "speeches-testimony":   ["interviews", "bsu-speeches", "bsu-interviews"],
    "research-surveillance": ["economic-bulletin", "macroeconomic-projections",
                              "research-bulletin"],
    "stability-regulation": ["financial-stability-review", "ssm-annual-report",
                             "macroprudential-bulletin", "ssm-supervision-newsletter",
                             "ecb-annual-report"],
}

# The press-conference page carries the statement AND the Q&A transcript.
# The Q&A section begins either at the ECB's own heading (all vintages since
# 1998 use a variant of it) or, on recent pages, right after the closing line
# of the statement. Split conservatively: no marker -> no split.
_QA_MARKERS = (
    "Transcript of the questions asked",          # 1998- : Q&A heading
    "We are now ready to take your questions.",   # 2021- : last statement line
    "We are now at your disposal for questions.", # older closing variant
)

def split_qa(text):
    """Return (statement, qa) -- qa is None when no Q&A section is found."""
    best = None
    for m in _QA_MARKERS:
        i = text.find(m)
        if i == -1:
            continue
        if m.startswith("We are now"):            # marker belongs to the statement
            i += len(m)
        if best is None or i < best:
            best = i
    if best is None or best < 200:                # too early to be a real split
        return text, None
    stmt = text[:best].strip()
    qa = text[best:].strip().lstrip("* \n")       # drop the "* * *" separator
    return (stmt, qa) if len(qa) > 200 else (text, None)

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
            raw = get_text(url)
            if not raw.strip():
                raise ValueError("no text extracted")
            with open(rpath, "w") as f:
                f.write(raw)
            cleaned = clean(raw)
            with open(cpath, "w") as f:
                f.write(cleaned)
            status = "ok"
            if source == "press-conference-statements":
                stmt, qa = split_qa(cleaned)
                if qa:
                    stem = cpath[:-len(".txt")]
                    with open(f"{stem}-stmt.txt", "w") as f:
                        f.write(stmt)
                    with open(f"{stem}-qa.txt", "w") as f:
                        f.write(qa)
                else:
                    status = "ok-no-qa-split"   # early meetings / odd layout
            print("y"); mw.writerow([datestr, url, cpath, status])
        except Exception as e: # pylint: disable=broad-except
            print("n"); ew.writerow([datestr, url, f"{type(e).__name__}: {e}"])
        time.sleep(1)

    manifest.close(); errorfile.close()

if __name__ == "__main__":
    args = sys.argv[1:]
    list_only = "--list" in args
    args = [a for a in args if a != "--list"] or list(SOURCES)
    targets = []
    for a in args:
        targets += GROUPS.get(a, [a])
    for src in dict.fromkeys(targets):
        if src not in SOURCES:
            raise SystemExit(f"unknown source '{src}'; choose from: "
                             f"{', '.join(list(SOURCES) + list(GROUPS))}")
        if list_only:   # discovery only: verify counts/patterns, download nothing
            items = get_index(src)
            span = f"{items[0][1]} .. {items[-1][1]}" if items else "EMPTY"
            print(f"[{src}] {len(items)} documents  {span}")
        else:
            scrape(src)
