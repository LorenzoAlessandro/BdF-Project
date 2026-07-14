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

_NAV = re.compile(r"Home\s+Media\s+Explainers\s+Research.*?Careers", re.S)
_CUTS = ("Thank you for letting us know", "We use functional cookies",
         "We are always working to improve this website", "All pages in this section")

def clean(text):
    """Trim ECB page furniture (nav bar, cookie/consent footer) and tidy
    whitespace while PRESERVING paragraph breaks. Conservative -- tune as needed."""
    cut = min([text.find(m) for m in _CUTS if m in text] + [len(text)])
    text = _NAV.sub("", text[:cut])
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# Paper slices -> scraper sources, so `python scrape.py deliberation-proxy`
# grabs the whole Table 3 row in one go.
GROUPS = {
    "deliberation-proxy": ["press-conference-statements", "monetary-policy-accounts"],
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
    stmt, qa = text[:best].strip(), text[best:].strip()
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
    args = sys.argv[1:] or list(SOURCES)
    targets = []
    for a in args:
        targets += GROUPS.get(a, [a])
    for src in dict.fromkeys(targets):
        scrape(src)
