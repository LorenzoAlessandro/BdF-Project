import os
import re
import time
import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ECB = "https://www.ecb.europa.eu"
BSU = "https://www.bankingsupervision.europa.eu"
CSV_DIR = "out/csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ecb-corpus-scraper/1.0; research use)"}

# One entry per document set. Each ECB section is built on the same template:
# a JS index whose data-snippets attribute lists per-year fragments, and pages
# addressed by an id in the filename. 'pattern' captures that id; 'id_kind' says
# how to read it; 'label' is the output-file suffix. Leading slash in a pattern
# anchors the filename so unrelated releases (e.g. ecb.pr...) don't match.
SOURCES = {
    "monetary-policy-decisions": dict(
        base=ECB, id_kind="date", label="mp", first_year=1999,
        index=f"{ECB}/press/govcdec/mopo/html/index.en.html",
        year_include=ECB + "/press/govcdec/mopo/{year}/html/index_include.en.html",
        pattern=r"/(?:ecb\.mp|pr)(\d{6})(?:~[0-9a-z]+)?\.en\.html"),
    "press-conference-statements": dict(
        base=ECB, id_kind="date", label="is", first_year=1998,
        index=f"{ECB}/press/press_conference/monetary-policy-statement/html/index.en.html",
        year_include=ECB + "/press/press_conference/monetary-policy-statement/{year}/html/index_include.en.html",
        pattern=r"/(?:ecb\.is|is)(\d{6})(?:~[0-9a-z]+)?\.en\.html"),
    "monetary-policy-accounts": dict(
        base=ECB, id_kind="date", label="account", first_year=2015,
        index=f"{ECB}/press/accounts/html/index.en.html",
        year_include=ECB + "/press/accounts/{year}/html/index_include.en.html",
        pattern=r"/(?:ecb\.mg|mg)(\d{6})(?:~[0-9a-z]+)?\.en\.html"),
    "economic-bulletin": dict(
        base=ECB, id_kind="yearissue", label="eb", first_year=2015, issues=8,
        index=f"{ECB}/press/economic-bulletin/html/index.en.html",
        year_include=ECB + "/press/economic-bulletin/{year}/html/index_include.en.html",
        pattern=r"/eb(\d{6})\.en\.html",
        # issues are predictable (no hash), so we can build the URLs directly
        construct=ECB + "/press/economic-bulletin/html/eb{year}{issue:02d}.en.html"),
    "macroeconomic-projections": dict(
        base=ECB, id_kind="yearmonth", label="proj", first_year=2000,
        index=f"{ECB}/press/projections/html/index.en.html",
        year_include=ECB + "/press/projections/{year}/html/index_include.en.html",
        pattern=r"/ecb\.projections(\d{6})_[a-z]+staff(?:~[0-9a-z]+)?\.en\.html"),
    "financial-stability-review": dict(
        base=ECB, id_kind="yearmonth", label="fsr", first_year=2004,
        index=f"{ECB}/press/financial-stability-publications/fsr/html/index.en.html",
        year_include=ECB + "/press/financial-stability-publications/fsr/{year}/html/index_include.en.html",
        pattern=r"/ecb\.fsr(\d{6})(?:~[0-9a-z]+)?\.en\.html"),
    "ssm-annual-report": dict(
        base=BSU, id_kind="year", label="ssm-ar", first_year=2014,
        index=f"{BSU}/press/other-publications/annual-report/html/index.en.html",
        year_include=BSU + "/press/other-publications/annual-report/{year}/html/index_include.en.html",
        pattern=r"/ssm\.ar(\d{4})~[0-9a-z]+\.en\.html"),
}

def get(url, tries=3):
    """GET with a real User-Agent, a timeout, and retries -- but do NOT retry
    4xx (e.g. 404), so probing non-existent fragments fails fast."""
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code < 500:
                raise                       # client error: pointless to retry
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))

def head_ok(url):
    """True if a URL exists (200), used to weed out not-yet-published issues."""
    try:
        return requests.head(url, headers=HEADERS, timeout=30,
                             allow_redirects=True).status_code == 200
    except requests.RequestException:
        return False

def parse_id(id_str, kind):
    """Return (sort_key, datestr) for an id, according to its scheme.
    sort_key[0] is always the year (used for the output folder)."""
    if kind == "date":            # YYMMDD
        yy, mm, dd = int(id_str[:2]), int(id_str[2:4]), int(id_str[4:6])
        year = 2000 + yy if yy < 90 else 1900 + yy
        return (year, mm, dd), f"{year}-{mm:02d}-{dd:02d}"
    if kind == "yearmonth":       # YYYYMM
        year, mm = int(id_str[:4]), int(id_str[4:6])
        return (year, mm, 0), f"{year}-{mm:02d}"
    if kind == "yearissue":       # YYYYNN (issue number)
        year, nn = int(id_str[:4]), int(id_str[4:6])
        return (year, nn, 0), f"{year}-i{nn:02d}"
    if kind == "year":            # YYYY
        year = int(id_str)
        return (year, 0, 0), f"{year}"
    raise ValueError(kind)

def find_links(html, rx, base_url):
    """id_str -> absolute url for every matching link in a page/fragment."""
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    for a in soup.find_all("a", href=True):
        m = rx.search(a["href"])
        if m:
            out[m.group(1)] = urljoin(base_url, a["href"])
    return out

def _snippet_urls(index_url):
    """Read the list of per-year fragment files off the index page (if any)."""
    soup = BeautifulSoup(get(index_url).text, "html.parser")
    el = soup.find(attrs={"data-snippets": True})
    if not el:
        return []
    return [urljoin(index_url, s.strip())
            for s in el["data-snippets"].split(",") if s.strip()]

def get_index(source):
    """Every document for a source as (sort_key, datestr, url), newest last.
    Merges three discovery routes so it works whichever one a section uses:
      (1) per-year fragments listed in data-snippets (or constructed);
      (2) links printed on the index page itself (how FSR exposes editions);
      (3) predictable constructed URLs for hash-less sets (economic bulletin)."""
    cfg = SOURCES[source]
    rx = re.compile(cfg["pattern"], re.I)
    this_year = datetime.date.today().year
    idmap = {}

    # (1) data-snippets fragments, else the constructed per-year fragment list
    try:
        fragments = _snippet_urls(cfg["index"])
    except requests.RequestException:
        fragments = []
    if not fragments:
        fragments = [cfg["year_include"].format(year=y)
                     for y in range(cfg["first_year"], this_year + 2)]
    for u in fragments:
        try:
            idmap.update(find_links(get(u).text, rx, u))
        except requests.RequestException:
            continue

    # (2) the index page frequently lists recent editions inline
    try:
        idmap.update(find_links(get(cfg["index"]).text, rx, cfg["index"]))
    except requests.RequestException:
        pass

    # (3) predictable back-catalogue: past years are certain, current year checked
    if cfg.get("construct"):
        for y in range(cfg["first_year"], this_year + 1):
            for n in range(1, cfg.get("issues", 8) + 1):
                key = f"{y}{n:02d}"
                if key in idmap:
                    continue
                url = cfg["construct"].format(year=y, issue=n)
                if y < this_year or head_ok(url):
                    idmap[key] = url

    items = []
    for id_str, url in idmap.items():
        try:
            key, datestr = parse_id(id_str, cfg["id_kind"])
        except (ValueError, IndexError):
            continue
        items.append((key, datestr, url))
    return sorted(items)

def raw_path(source, year, datestr):
    return f"raw/{source}/{year}/{datestr}-{SOURCES[source]['label']}.txt"

def clean_path(source, year, datestr):
    return f"clean/{source}/{year}/{datestr}-{SOURCES[source]['label']}.txt"
