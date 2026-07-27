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
            with open(cpath, "w") as f:
                f.write(clean(raw))
            print("y"); mw.writerow([datestr, url, cpath, "ok"])
        except Exception as e: # pylint: disable=broad-except
            print("n"); ew.writerow([datestr, url, f"{type(e).__name__}: {e}"])
        time.sleep(1)

    manifest.close(); errorfile.close()

if __name__ == "__main__":
    targets = sys.argv[1:] or list(SOURCES)
    for src in targets:
        scrape(src)
