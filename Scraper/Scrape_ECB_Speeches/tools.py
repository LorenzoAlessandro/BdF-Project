import os
import re
import time
import datetime
from urllib.parse import urljoin, unquote

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
    "speeches": dict(
        # One page per speech, all under /press/key/date/{year}/html/. Several
        # speeches can share a date, so the id is YYMMDD with an optional _N
        # suffix (id_kind="dateseq"). Two URL generations coexist, mirroring
        # the press conferences:
        #   1997-2019  .../key/date/{year}/html/spYYMMDD[_N].en.html
        #   2019-      .../key/date/{year}/html/ecb.spYYMMDD[_N]~hash.en.html
        # The archive starts in 1997 (a few EMI-era speeches precede the ECB).
        base=ECB, id_kind="dateseq", label="sp", first_year=1997,
        # NOTE: this index URL now redirects to the JS-filtered "Publications
        # by date" app (/press/pubbydate/...). Its data-snippets, when
        # present, list fragments covering every publication type; the
        # sp-pattern below keeps speeches only. The per-year listing pages
        # under /press/key/date/{year}/ are still served statically, so they
        # are added as an extra fragment template for robustness.
        index=f"{ECB}/press/key/html/index.en.html",
        year_include=[ECB + "/press/key/date/{year}/html/index_include.en.html",
                      ECB + "/press/key/date/{year}/html/index.en.html",
                      ECB + "/press/key/{year}/html/index_include.en.html"],
        pattern=r"/(?:ecb\.sp|sp)(\d{6}(?:_\d+)?)(?:~[0-9a-z]+)?\.en\.html"),
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
    if kind == "dateseq":         # YYMMDD or YYMMDD_N (nth speech that day)
        stem, _, seq = id_str.partition("_")
        (year, mm, dd), datestr = parse_id(stem, "date")
        n = int(seq) if seq else 0
        return (year, mm, dd, n), (f"{datestr}-{n}" if n else datestr)
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
                if key in idmap:
                    continue
                url = cfg["construct"].format(year=y, issue=n)
                if y < this_year or head_ok(url):
                    idmap[key] = url

    # (4) gap-fill years with no hits yet by probing every fragment template.
    # 404s raise HTTPError from get() (no retry on 4xx) and are skipped, so a
    # missing template generation costs one fast request per year at most.
    have_years = {y for y in (_id_year(i, cfg["id_kind"]) for i in idmap)
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

    items = []
    for id_str, url in idmap.items():
        try:
            key, datestr = parse_id(id_str, cfg["id_kind"])
        except (ValueError, IndexError):
            continue
        items.append((key, datestr, url))
    return sorted(items)

# --- topic tags -------------------------------------------------------------
# The website's speeches index redirects to the "Publications by date" app,
# which filters CLIENT-SIDE on URL parameters such as
#   ?name_of_publication=Speech&topic=Monetary%20policy
# so the filtered URL itself returns no server-rendered list to requests.
# The tags powering that filter are not printed on the article pages either,
# so we harvest them from the app's own per-year fragments (found via the
# same data-snippets mechanism used by get_index) and key them by document id.

PUBBYDATE = f"{ECB}/press/pubbydate/html/index.en.html"

def _entry_topics(entry):
    """Best-effort topic harvest from one list entry, covering the carriers
    the ECB templates use for filter metadata: ?topic= links, elements whose
    class mentions topic/taxonomy/tag, and data-*topic* attributes."""
    out = []
    for a in entry.find_all("a", href=True):
        m = re.search(r"[?&]topic=([^&#]+)", a["href"])
        if m:
            out.append(unquote(m.group(1)).replace("+", " "))
    for el in entry.find_all(class_=re.compile(r"topic|taxonom|tag", re.I)):
        out += el.get_text("|", strip=True).split("|")
    for el in [entry] + entry.find_all(True):
        for k, v in el.attrs.items():
            if "topic" in k.lower() and isinstance(v, str):
                out += re.split(r"[|,;]", v)
    seen = []
    for t in (t.strip() for t in out):
        if t and t not in seen:
            seen.append(t)
    return seen

def get_topics(source):
    """id_str -> [topic tags] for a source, read from the fragments behind
    the Publications-by-date filter. An empty dict means the tags could not
    be found (treat as 'filter unavailable', NOT as 'nothing matches')."""
    cfg = SOURCES[source]
    rx = re.compile(cfg["pattern"], re.I)
    this_year = datetime.date.today().year
    try:
        fragments = _snippet_urls(PUBBYDATE)
    except requests.RequestException:
        fragments = []
    if not fragments:
        fragments = [f"{ECB}/press/pubbydate/{y}/html/index_include.en.html"
                     for y in range(cfg["first_year"], this_year + 2)]
    topics = {}
    for u in dict.fromkeys(fragments):
        try:
            soup = BeautifulSoup(get(u).text, "html.parser")
        except requests.RequestException:
            continue
        for a in soup.find_all("a", href=True):
            m = rx.search(a["href"])
            if not m:
                continue
            entry = a.find_parent("dd") or a.parent
            for t in _entry_topics(entry):
                topics.setdefault(m.group(1), [])
                if t not in topics[m.group(1)]:
                    topics[m.group(1)].append(t)
    return topics

def raw_path(source, year, datestr):
    return f"raw/{source}/{year}/{datestr}-{SOURCES[source]['label']}.txt"

def clean_path(source, year, datestr):
    return f"clean/{source}/{year}/{datestr}-{SOURCES[source]['label']}.txt"
