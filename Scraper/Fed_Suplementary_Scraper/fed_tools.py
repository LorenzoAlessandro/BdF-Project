"""Fed counterpart to the ECB tools.py -- same contract, different discovery.

The Fed site has no data-snippets JS index, so discovery is per-source:
static per-year index pages (testimony, FEDS notes), the FOMC calendar +
fomchistorical{YYYY}.htm pages (statements / press conferences /
projections), and one flat index page (annual report).

get_index(source) returns (sort_key, datestr, url) items, newest last,
exactly like the ECB version, so fed_scrape.py mirrors scrape.py.

URL patterns below were verified against federalreserve.gov on 2026-07-16.

Corpus mapping (ECB branch -> this file):
  annual_report_archive        -> annual-report          (PDF, 1995-)
  other_gc_decisions           -> fomc-statements        (1994-; includes
                                  implementation notes 'a1' and extra
                                  releases 'b'/'c' e.g. longer-run goals)
  press_conference_statements  -> fomc-press-conferences (transcript PDFs, 2011-)
  research_bulletin            -> feds-notes              (2013-)
  mep_letters                  -> testimony               (2006-; pre-2006
                                  lives under /boarddocs/testimony, not covered)
  macroeconomic-projections    -> fomc-projections        (SEP tables, bonus,
                                  for parity with the ECB scraper config)
"""

import re
import time
import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

FED = "https://www.federalreserve.gov"
CSV_DIR = "out/csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fed-corpus-scraper/1.0; research use)"}

# Each pattern uses named groups: (?P<d>...) = date-ish id, (?P<x>...) = an
# optional discriminator (statement suffix, speaker name, note slug) so that
# several documents on the same day get distinct keys and filenames.
# Leading '/' anchors the path so e.g. the statement *PDF* links
# (/monetarypolicy/files/monetaryYYYYMMDDa1.pdf) don't collide with the
# statement/implementation-note HTML pages.
SOURCES = {
    # Post-meeting policy statements ('a'), implementation notes ('a1') and
    # companion releases like the Statement on Longer-Run Goals ('b','c').
    # Modern era 2006- : /newsevents/pressreleases/monetaryYYYYMMDD{a,a1,b,c}.htm
    # 1996-2005        : /boarddocs/press/monetary/YYYY/YYYYMMDD/  (also /general/)
    # NOTE 1994-95 statements use yet older paths; check the first-run counts
    # and add a pattern if those two years matter to you.
    "fomc-statements": dict(
        kind="fomc", label="stmt", first_year=1994,
        patterns=[
            r"/newsevents/pressreleases/monetary(?P<d>\d{8})(?P<x>[a-z]\d?)\.htm",
            r"/boarddocs/press/(?:monetary|general)/\d{4}/(?P<d>\d{8})/",
        ]),
    # Post-meeting press conference transcripts (Chair Q&A), April 2011-.
    # Dates are discovered from the calendar/historical pages (the page id
    # is occasionally spelled 'fomcpressconf', hence press?), and the
    # transcript itself is a predictable PDF under /mediacenter/files/.
    "fomc-press-conferences": dict(
        kind="fomc", label="presconf", first_year=2011,
        patterns=[r"/monetarypolicy/fomcpress?conf(?P<d>\d{8})\.htm"],
        doc_url=FED + "/mediacenter/files/FOMCpresconf{d}.pdf"),
    # Summary of Economic Projections tables (one 2022 page is spelled
    # 'fomcprojtable', hence the optional 'e').
    "fomc-projections": dict(
        kind="fomc", label="proj", first_year=2011,
        patterns=[r"/monetarypolicy/fomcprojtabl(?:e)?(?P<d>\d{8})\.htm"]),
    # Congressional testimony by Board members and senior staff.
    "testimony": dict(
        kind="years", label="testimony", first_year=2006,
        year_index=FED + "/newsevents/testimony/{year}-testimony.htm",
        patterns=[r"/newsevents/testimony/(?P<x>[a-z-]+)(?P<d>\d{8})(?P<y>[a-z])\.htm"]),
    # FEDS Notes (staff research notes), 2013-. Article pages end .html.
    "feds-notes": dict(
        kind="years", label="fedsnote", first_year=2013,
        year_index=FED + "/econres/notes/feds-notes/{year}-index.htm",
        all_years=FED + "/econres/notes/feds-notes/all-years.htm",
        patterns=[r"/econres/notes/feds-notes/(?P<x>[A-Za-z0-9-]+?)-(?P<d>\d{8})\.html?"]),
    # Board of Governors Annual Report, one PDF per year, 1995-.
    # Filenames are irregular across eras (2024-annual-report.pdf, AR09.pdf,
    # ar04.pdf, ann99.pdf, annual96/annual.pdf) so the year is pulled from
    # the href rather than constructed.
    "annual-report": dict(
        kind="annual", label="ar", first_year=1995,
        index=FED + "/publications/annual-report.htm"),
}


def get(url, tries=3):
    """GET with a real User-Agent, a timeout, and retries -- but do NOT retry
    4xx (e.g. 404), so probing not-yet-published year pages fails fast."""
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
    """True if a URL exists (200)."""
    try:
        return requests.head(url, headers=HEADERS, timeout=30,
                             allow_redirects=True).status_code == 200
    except requests.RequestException:
        return False


def _find_ids(html, patterns, base_url):
    """(date8, disc) -> absolute url for every matching link in a page."""
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    for a in soup.find_all("a", href=True):
        for pat in patterns:
            m = pat.search(a["href"])
            if m:
                g = m.groupdict()
                disc = (g.get("x") or "") + (g.get("y") or "")
                out[(g["d"], disc.lower())] = urljoin(base_url, a["href"])
                break
    return out


def _scan_pages(urls, patterns, quiet_404=True):
    """Union of _find_ids over many index pages, skipping missing ones."""
    idmap = {}
    for u in urls:
        try:
            idmap.update(_find_ids(get(u).text, patterns, u))
        except requests.HTTPError as e:
            if not (quiet_404 and e.response is not None
                    and e.response.status_code == 404):
                print(f"  ! {u}: {e}")
        except requests.RequestException as e:
            print(f"  ! {u}: {e}")
        time.sleep(0.5)
    return idmap


def _fomc_index_pages(first_year):
    """Where FOMC meeting materials are listed: the rolling calendar page
    (current + ~5 previous years) plus one fomchistorical page per year.
    Historical pages appear with a ~5-year lag; 404s on recent years are
    expected and skipped, so the two routes always overlap rather than gap."""
    this_year = datetime.date.today().year
    pages = [f"{FED}/monetarypolicy/fomccalendars.htm"]
    pages += [f"{FED}/monetarypolicy/fomchistorical{y}.htm"
              for y in range(first_year, this_year + 1)]
    return pages


def _year_from_ar_href(href):
    """Pull the report year out of an annual-report PDF link, across eras:
    2024-annual-report.pdf | annual09/pdf/AR09.pdf | annual04/ar04.pdf |
    annual99/ann99.pdf | annual96/annual.pdf | ann95.pdf"""
    m = re.search(r"(\d{4})-annual-report\.pdf", href, re.I)
    if m:
        return int(m.group(1))
    for rx in (r"annual(\d{2})/", r"\bann(\d{2})\.pdf", r"\bar(\d{2})\.pdf"):
        m = re.search(rx, href, re.I)
        if m:
            yy = int(m.group(1))
            return 1900 + yy if yy >= 90 else 2000 + yy
    return None


def get_index(source):
    """Every document for a source as (sort_key, datestr, url), newest last.
    sort_key[0] is always the year (used for the output folder)."""
    cfg = SOURCES[source]
    this_year = datetime.date.today().year
    items = []

    if cfg["kind"] == "fomc":
        pats = [re.compile(p, re.I) for p in cfg["patterns"]]
        idmap = _scan_pages(_fomc_index_pages(SOURCES["fomc-statements"]["first_year"]), pats)
        for (d, disc), url in idmap.items():
            y, m, dd = int(d[:4]), int(d[4:6]), int(d[6:8])
            if y < cfg["first_year"]:
                continue
            if cfg.get("doc_url"):                       # e.g. transcript PDF
                url = cfg["doc_url"].format(d=d)
            datestr = f"{y}-{m:02d}-{dd:02d}" + (f"-{disc}" if disc else "")
            items.append(((y, m, dd, disc), datestr, url))

    elif cfg["kind"] == "years":
        pats = [re.compile(p, re.I) for p in cfg["patterns"]]
        # prefer the site's own list of year indexes when it has one
        pages = []
        if cfg.get("all_years"):
            try:
                soup = BeautifulSoup(get(cfg["all_years"]).text, "html.parser")
                pages = [urljoin(cfg["all_years"], a["href"])
                         for a in soup.find_all("a", href=True)
                         if re.search(r"\d{4}-index\.htm", a["href"])]
            except requests.RequestException:
                pages = []
        if not pages:
            pages = [cfg["year_index"].format(year=y)
                     for y in range(cfg["first_year"], this_year + 1)]
        idmap = _scan_pages(pages, pats)
        for (d, disc), url in idmap.items():
            y, m, dd = int(d[:4]), int(d[4:6]), int(d[6:8])
            disc = re.sub(r"-+$", "", disc[:60])         # keep filenames sane
            datestr = f"{y}-{m:02d}-{dd:02d}" + (f"-{disc}" if disc else "")
            items.append(((y, m, dd, disc), datestr, url))

    elif cfg["kind"] == "annual":
        soup = BeautifulSoup(get(cfg["index"]).text, "html.parser")
        seen = {}
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.lower().endswith(".pdf"):
                continue
            year = _year_from_ar_href(href)
            if year and year not in seen:                # page lists newest first
                seen[year] = urljoin(cfg["index"], href)
        for year, url in seen.items():
            items.append(((year, 0, 0, ""), f"{year}", url))

    else:
        raise ValueError(cfg["kind"])

    return sorted(items)


def raw_path(source, year, datestr):
    return f"raw/{source}/{year}/{datestr}-{SOURCES[source]['label']}.txt"


def clean_path(source, year, datestr):
    return f"clean/{source}/{year}/{datestr}-{SOURCES[source]['label']}.txt"
