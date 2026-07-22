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
        # One page per Governing Council meeting: the introductory / monetary
        # policy statement WITH the Q&A transcript appended (the ECB publishes
        # them together). Two URL generations coexist:
        #   1998-2021  /press/pressconf/{year}/html/isYYMMDD.en.html
        #   2021-      /press/press_conference/monetary-policy-statement/
        #              {year}/html/ecb.isYYMMDD~hash.en.html
        # so we search both the new section and the legacy one.
        base=ECB, id_kind="date", label="is", first_year=1998,
        index=[f"{ECB}/press/press_conference/monetary-policy-statement/html/index.en.html",
               f"{ECB}/press/pressconf/html/index.en.html"],
        year_include=[ECB + "/press/press_conference/monetary-policy-statement/{year}/html/index_include.en.html",
                      ECB + "/press/pressconf/{year}/html/index_include.en.html"],
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
        # NOTE: this source has NO per-year subfolder (ssm.ar2015.en.html and
        # ssm.ar2022~hash.en.html both live directly under .../html/, unlike
        # most other sections) -- so year_include's {year} guess never
        # matches anything. The real full listing lives on all-releases.
        index=[f"{BSU}/press/other-publications/annual-report/html/index.en.html",
               f"{BSU}/press/other-publications/annual-report/html/all-releases.en.html"],
        year_include=BSU + "/press/other-publications/annual-report/{year}/html/index_include.en.html",
        # hash optional: 2015/2016 reports are named 'ssm.ar2015.en.html'
        # with no trailing hash, unlike later years.
        pattern=r"/ssm\.ar(\d{4})(?:~[0-9a-z]+)?\.en\.html"),
    # ------------------------------------------------------------------
    # Additional sources per paper slice. URL patterns marked [verified]
    # were confirmed against live pages; [best-effort] follow the standard
    # ECB naming template but should be sanity-checked with
    # `python scrape.py --list <source>` before a full download.
    # ------------------------------------------------------------------
    "other-gc-decisions": dict(          # [best-effort] slice: policy decisions
        base=ECB, id_kind="date", label="gc", first_year=1999,
        index=f"{ECB}/press/govcdec/otherdec/html/index.en.html",
        year_include=ECB + "/press/govcdec/otherdec/{year}/html/index_include.en.html",
        pattern=r"/(?:ecb\.gc|gc)(\d{6})(?:~[0-9a-z]+)?\.en\.html"),
    "interviews": dict(                  # [verified: ecb.inYYMMDD~hash] slice: speeches & testimony
        # Q&A-format press contributions by Executive Board members. Per-year
        # pages here are the full 'index.en.html' (this section never used
        # the '_include' fragment convention other sources use).
        base=ECB, id_kind="date", label="in", first_year=1999,
        index=f"{ECB}/press/inter/html/index.en.html",
        year_include=ECB + "/press/inter/date/{year}/html/index.en.html",
        pattern=r"/(?:ecb\.in|in|sp)(\d{6})(?:_\d+)?(?:~[0-9a-z]+)?\.en\.html"),
    "research-bulletin": dict(           # [best-effort] slice: research & surveillance
        base=ECB, id_kind="date", label="rb", first_year=2015,
        index=f"{ECB}/press/research-publications/resbull/html/index.en.html",
        year_include=ECB + "/press/research-publications/resbull/{year}/html/index_include.en.html",
        pattern=r"/ecb\.rb(\d{6}(?:_\d+)?)(?:~[0-9a-z]+)?\.en\.html"),
    "macroprudential-bulletin": dict(    # [best-effort] slice: stability & regulation
        base=ECB, id_kind="yearmonth", label="mpb", first_year=2016,
        index=f"{ECB}/press/financial-stability-publications/macroprudential-bulletin/html/index.en.html",
        year_include=ECB + "/press/financial-stability-publications/macroprudential-bulletin/{year}/html/index_include.en.html",
        pattern=r"/ecb\.mpbu(\d{6}(?:_\d+)?)(?:~[0-9a-z]+)?\.en\.html"),
    "ecb-annual-report": dict(           # [best-effort] HTML era; 1998-2014 via scrape_pdf.py
        base=ECB, id_kind="year", label="ar", first_year=2014,
        index=f"{ECB}/press/annual-reports-financial-statements/annual/html/index.en.html",
        year_include=ECB + "/press/annual-reports-financial-statements/annual/{year}/html/index_include.en.html",
        pattern=r"/ecb\.ar(\d{4})(?:_\d+)?(?:~[0-9a-z]+)?\.en\.html"),
    "ssm-supervision-newsletter": dict(  # [best-effort] slice: stability & regulation
        base=BSU, id_kind="date", label="ssm-nl", first_year=2017,
        index=f"{BSU}/press/publications/newsletter/html/index.en.html",
        year_include=BSU + "/press/publications/newsletter/{year}/html/index_include.en.html",
        pattern=r"/ssm\.nl(\d{6}(?:_\d+)?)(?:~[0-9a-z]+)?\.en\.html"),
    "bsu-speeches": dict(                # [best-effort] Supervisory Board speeches
        base=BSU, id_kind="date", label="ssm-sp", first_year=2014,
        index=f"{BSU}/press/speeches/html/index.en.html",
        year_include=BSU + "/press/speeches/date/{year}/html/index_include.en.html",
        pattern=r"/ssm\.sp(\d{6}(?:_\d+)?)(?:~[0-9a-z]+)?\.en\.html"),
    "bsu-interviews": dict(              # [best-effort] Supervisory Board interviews
        base=BSU, id_kind="date", label="ssm-in", first_year=2014,
        index=f"{BSU}/press/interviews/html/index.en.html",
        year_include=BSU + "/press/interviews/date/{year}/html/index_include.en.html",
        pattern=r"/ssm\.in(\d{6}(?:_\d+)?)(?:~[0-9a-z]+)?\.en\.html"),
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
    sort_key[0] is always the year (used for the output folder). An id may
    carry an article suffix '_N' (bulletins/newsletters publish several
    articles per issue); it is kept so items stay distinct and sortable."""
    base, _, part = id_str.partition("_")
    part = int(part) if part.isdigit() else 0
    suffix = f"-a{part:02d}" if part else ""
    if kind == "date":            # YYMMDD
        yy, mm, dd = int(base[:2]), int(base[2:4]), int(base[4:6])
        year = 2000 + yy if yy < 90 else 1900 + yy
        return (year, mm, dd, part), f"{year}-{mm:02d}-{dd:02d}{suffix}"
    if kind == "yearmonth":       # YYYYMM
        year, mm = int(base[:4]), int(base[4:6])
        return (year, mm, 0, part), f"{year}-{mm:02d}{suffix}"
    if kind == "yearissue":       # YYYYNN (issue number)
        year, nn = int(base[:4]), int(base[4:6])
        return (year, nn, 0, part), f"{year}-i{nn:02d}{suffix}"
    if kind == "year":            # YYYY
        year = int(base)
        return (year, 0, 0, part), f"{year}{suffix}"
    raise ValueError(kind)

def find_links(html, rx, base_url):
    """full-matched-filename -> (id_str, absolute url) for every matching
    link. Keyed on the full match (hash included) so that two documents
    sharing a date -- e.g. two interviews on the same day -- both survive."""
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    for a in soup.find_all("a", href=True):
        m = rx.search(a["href"])
        if m:
            out[m.group(0)] = (m.group(1), urljoin(base_url, a["href"]))
    return out

def _snippet_urls(index_url):
    """Read the list of per-year fragment files off the index page (if any)."""
    soup = BeautifulSoup(get(index_url).text, "html.parser")
    el = soup.find(attrs={"data-snippets": True})
    if not el:
        return []
    return [urljoin(index_url, s.strip())
            for s in el["data-snippets"].split(",") if s.strip()]

def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]

def _id_year(id_str, kind):
    try:
        return parse_id(id_str, kind)[0][0]
    except (ValueError, IndexError):
        return None

def get_index(source):
    """Every document for a source as (sort_key, datestr, url), newest last.
    Merges four discovery routes so it works whichever one a section uses:
      (1) per-year fragments listed in data-snippets (or constructed) -- for
          sources with several URL generations, ALL index sections are read;
      (2) links printed on the index page(s) themselves;
      (3) predictable constructed URLs for hash-less sets (economic bulletin);
      (4) gap-fill: any year >= first_year that is still empty is probed
          directly against every year_include template, which is what pulls
          in the 1998-2001 press conferences if the JS index omits them."""
    cfg = SOURCES[source]
    rx = re.compile(cfg["pattern"], re.I)
    this_year = datetime.date.today().year
    indexes = _as_list(cfg["index"])
    templates = _as_list(cfg["year_include"])
    idmap = {}

    # (1) data-snippets fragments, else the constructed per-year fragment list
    fragments = []
    for idx in indexes:
        try:
            fragments += _snippet_urls(idx)
        except requests.RequestException:
            continue
    if not fragments:
        fragments = [t.format(year=y) for t in templates
                     for y in range(cfg["first_year"], this_year + 2)]
    for u in dict.fromkeys(fragments):          # dedupe, keep order
        try:
            idmap.update(find_links(get(u).text, rx, u))
        except requests.RequestException:
            continue
        time.sleep(0.25)

    # (2) the index pages frequently list recent editions inline
    for idx in indexes:
        try:
            idmap.update(find_links(get(idx).text, rx, idx))
        except requests.RequestException:
            pass

    # (3) predictable back-catalogue: past years are certain, current year checked
    if cfg.get("construct"):
        for y in range(cfg["first_year"], this_year + 1):
            for n in range(1, cfg.get("issues", 8) + 1):
                key = f"{y}{n:02d}"
                if any(v[0] == key for v in idmap.values()):
                    continue
                url = cfg["construct"].format(year=y, issue=n)
                if y < this_year or head_ok(url):
                    idmap[url] = (key, url)

    # (4) gap-fill years with no hits yet by probing every fragment template,
    # plus the site-wide publications-by-date fragment as a last resort (it
    # lists everything published that year; our pattern picks out this
    # source's documents). 404s raise from get() (no retry on 4xx) and are
    # skipped, so a missing template generation costs one fast request/year.
    templates = templates + [cfg["base"] + "/press/pubbydate/{year}/html/index_include.en.html"]
    # some sections never used the '_include' fragment convention at all;
    # trying the plain per-year page too costs one extra 404 in the common
    # case but recovers sources where the '_include' guess was simply wrong.
    templates += [t.replace("index_include.en.html", "index.en.html")
                  for t in templates if "index_include.en.html" in t]
    have_years = {y for y in (_id_year(v[0], cfg["id_kind"]) for v in idmap.values())
                  if y is not None}
    for y in range(cfg["first_year"], this_year + 1):
        if y in have_years:
            continue
        for t in templates:
            u = t.format(year=y)
            try:
                idmap.update(find_links(get(u).text, rx, u))
            except requests.RequestException:
                continue
            time.sleep(0.25)

    items = []
    for id_str, url in idmap.values():
        try:
            key, datestr = parse_id(id_str, cfg["id_kind"])
        except (ValueError, IndexError):
            continue
        items.append((key, datestr, url))
    items.sort()

    # two documents on the same date get -b, -c, ... so output paths differ
    seen, out = {}, []
    for key, datestr, url in items:
        n = seen.get(datestr, 0)
        seen[datestr] = n + 1
        if n:
            datestr = f"{datestr}-{chr(ord('a') + n)}"
        out.append((key, datestr, url))
    return out

def raw_path(source, year, datestr):
    return f"raw/{source}/{year}/{datestr}-{SOURCES[source]['label']}.txt"

def clean_path(source, year, datestr):
    return f"clean/{source}/{year}/{datestr}-{SOURCES[source]['label']}.txt"
