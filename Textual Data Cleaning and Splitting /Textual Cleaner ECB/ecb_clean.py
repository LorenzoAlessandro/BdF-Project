#!/usr/bin/env python3
"""
ecb_clean.py — Cleaning pipeline for scraped/extracted ECB texts -> LLM training data.

Handles two source families:
  A) Web-scraped pages  (statements, speeches, interviews, GC decisions,
     accounts, research bulletins, press material)
  B) PDF-extracted pubs (Annual Report, Monthly Bulletin, Financial Stability
     Review, letters to MEPs, bulletin articles)

Pipeline stages
  1. Unicode & PDF-artifact normalisation (ligatures, soft hyphens, NBSP)
  2. Head zone strip   (website navigation cruft)
  3. Tail zone strip   (feedback widget, CONTACT block, attendee rosters)
  4. TOC zone strip    (Contents ... first real heading)
  5. Line-level filters (running headers, page numbers, imprint, chart/table
     apparatus, letterheads, cover art, nav tokens)
  6. Block filters     (destroyed numeric tables, single-letter runs,
     name rosters, References sections)
  7. De-hyphenation    (corpus-vocabulary heuristic)
  8. Paragraph reflow  (join hard-wrapped lines; filtered lines never bridge)
  9. Paragraph scoring (drop residual non-narrative debris: citation dumps,
     figure stubs, low-alpha fragments)
 10. Sentence scrubs & inline fixes (footnote markers, ordinals, boilerplate
     sentences, whitespace)

Usage
  python ecb_clean.py IN_DIR OUT_DIR [--jsonl corpus.jsonl] [--report report.csv]
                                     [--keep-footnotes] [--min-words 8] [--debug]
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Configuration: pattern catalog                                             #
# --------------------------------------------------------------------------- #

HARD_BREAK = "\x00"          # sentinel: a filtered line leaves a paragraph break

MONTHS = ("January|February|March|April|May|June|July|August|September|"
          "October|November|December")

# --- exact standalone lines: website navigation / UI chrome ---------------- #
NAV_TOKENS = {
    "suggestions", "sort by", "relevance", "date", "anytime", "past month",
    "past year", "search options", "image preview", "skip to:", "skip to",
    "navigation", "content", "footer", "media contacts", "related topics",
    "yes", "no", "what made you unhappy?", "page not working",
    "information not useful", "design not attractive", "something else",
    "print", "share", "email", "twitter", "facebook", "linkedin",
    "disclaimer", "languages", "english", "select your language",
}

# --- tail cut markers: everything from here to EOF is chrome --------------- #
TAIL_CUT_RE = [
    re.compile(r"^Are you happy with this page\?", re.I),
    re.compile(r"^CONTACT$"),
    re.compile(r"^Media contacts$", re.I),
    re.compile(r"^Meeting of the ECB.s Governing Council, .*\d{4}$"),
]

# annex sections that may occupy a large tail share -> lower floor
ANNEX_CUT_RE = [
    re.compile(r"^STATISTICAL ANNEX$"),
    re.compile(r"^Documents published by the( European Central Bank)?$"),
    re.compile(r"^Documents published by the European Central Bank"),
]

# --- single-line filters (regex, applied to stripped line) ----------------- #
LINE_DROP_RE = [
    # running headers / footers
    re.compile(r"^ECB Annual Report\s*[•·]\s*\d{4}$"),
    re.compile(r"^ECB\s*[•·]\s*Monthly Bulletin\s*[•·]\s*\w+ \d{4}$"),
    re.compile(r"^Financial Stability Review( \w+ \d{4})?$", re.I),
    re.compile(r"^Monthly Bulletin$", re.I),
    re.compile(r"^European Central Bank$", re.I),
    re.compile(r"^Annual Report( \d{4})?$", re.I),
    re.compile(r"^ECB$"),
    re.compile(rf"^({MONTHS}) \d{{4}}$", re.I),
    # page numbers: arabic (opt. MB annex star), roman numerals
    re.compile(r"^\d{1,3}\s?\*?$"),
    re.compile(r"^[IVXLCDM]{1,7}$"),
    # imprint / copyright block
    re.compile(r"^©"),
    re.compile(r"^(Postal\s+)?[Aa]ddress$"),
    re.compile(r"^(Telephone|Telefax|Fax|Telex|Internet|Website|Tel)\b[.:]?"
               r"\s*\S{0,60}$"),
    re.compile(r"^\+\d[\d\s()/-]{6,}$"),
    re.compile(r"^(Kaiserstrasse|Sonnemannstrasse|Postfach)\b"),
    re.compile(r"^D?-?\d{5}\s+Frankfurt am Main$"),
    re.compile(r"^60\d{3} Frankfurt am Main$"),
    re.compile(r"^Frankfurt am Main$"),
    re.compile(r"^Germany$"),
    re.compile(r"^All rights reserved\.?"),
    re.compile(r"^Reproduction (is permitted|for educational)"),
    re.compile(r"^The cut-?off date for"),
    re.compile(r"^Unless otherwise stated, this document uses$"),
    re.compile(r"^data available as at"),
    re.compile(r"^ISSN\b|^ISBN\b|^DOI\b"),
    re.compile(r"^This Bulletin was produced under the responsibility"),
    re.compile(r"^the national central banks\.$"),
    re.compile(r"^Translations are prepared and published by$"),
    re.compile(r"^In \d{4} all ECB$"),
    re.compile(r"^publications$"),
    re.compile(r"^feature a motif$"),
    re.compile(r"^taken from the$"),
    re.compile(r"^€\d+ banknote\.$"),
    re.compile(r"^E ?N$"),
    # letter apparatus
    re.compile(r"^L/[A-Z]{1,4}/\d+/\d+$"),
    re.compile(r"^\[signed\]$", re.I),
    re.compile(r"^Member of the European Parliament$"),
    re.compile(r"^European Parliament$"),
    re.compile(r"^\d+,?\s+rue\s+\w+", re.I),
    re.compile(r"^B-\d{4}\s+\w+"),
    re.compile(r"^European Central Bank$"),
    # chart / table / figure apparatus
    re.compile(r"^(Chart|Table|Figure|Box)\s+[A-Z]?\d+[a-z]?\b"),
    re.compile(r"^Sources?:"),
    re.compile(r"^Notes?:"),
    re.compile(r"^\(.*\)$"),                      # whole-line parentheticals
    re.compile(r".*\((left|right)-hand scale\b.*", re.I),
    re.compile(r"^(x-axis|y-axis)\b", re.I),
    # standalone footnote definitions & markers
    re.compile(r"^\d{1,2}\)\s"),
    re.compile(r"^\[\d{1,2}\]$"),
    re.compile(r"^\*{1,3}\s"),
    # standalone urls / mails
    re.compile(r"^(https?://|www\.)\S+$"),
    re.compile(r"^\S+@\S+\.\S+$"),
    # scheduling boilerplate
    re.compile(r"^Release of the next monetary policy account"),
    # datelines
    re.compile(rf"^Frankfurt( am Main)?, (\d{{1,2}} )?({MONTHS}) \d{{4}}$"),
    # letter sender / recipient name & role lines
    re.compile(r"^[A-Z][a-zà-ÿ]+(?:\s[A-Z][a-zà-ÿ]+)?\s+[A-Z]{4,}$"),
    re.compile(r"^(President|Vice-President|Chair(?:person)?(?: of the "
               r"Supervisory Board)?|Executive Board Member)$"),
    re.compile(r"^(Mr|Ms|Mrs|Dr)\.?\s+[A-Z][A-Za-zà-ÿ]+\s+[A-Z][A-Za-zà-ÿ]+$"),
    re.compile(r"^Yours sincerely,?$"),
    # numbered section headings / TOC entries without terminal punctuation
    re.compile(r"^\d{1,2}(\.\d{1,2})?\s+[A-Z\u201c\"][^.?!]{0,90}$"),
    # FSR annex chart-list entries ending in S-page references
    re.compile(r"^.{0,90}\sS\d{1,3}$"),
    # annex headings for material we drop anyway
    re.compile(r"^Documents published by the European Central Bank"),
    re.compile(r"^Chronology of monetary policy measures"),
    re.compile(r"^Abbreviations$"),
    # doc-category caps tags (title/speaker/date lines are kept)
    re.compile(r"^(SPEECH|INTERVIEW|PRESS RELEASE|PRESS CONFERENCE|STATEMENT|"
               r"BLOG POST|PODCAST|THE ECB BLOG)$"),
]

# --- narrative-context lines to KEEP even though they look like headers ---- #
LINE_KEEP_RE = []

DATELINE_RE = re.compile(
    rf"^Frankfurt( am Main)?, (\d{{1,2}} )?({MONTHS}) \d{{4}}$")

# --- roster lines (attendee lists in accounts) ------------------------------ #
ROSTER_HEAD_RE = re.compile(
    r"^(Members|Other attendees|Accompanying persons|Other ECB staff|"
    r"Voting members|Non-voting members)$")
ROSTER_NAME_RE = re.compile(
    r"^(Mr|Ms|Mrs|Dr|Prof)\.?\s+[A-ZÀ-Þ][\w’'.\- ]{1,40}(,.*)?$")
ROSTER_NOTE_RE = re.compile(r"^\*{1,3}\s|^\*{1,3}$")

# --- references / bibliography ---------------------------------------------- #
REFS_HEAD_RE = re.compile(r"^(References|REFERENCES|Bibliography|Annexes)$")
CITE_SIG_RE = re.compile(
    r"\(\d{4}[a-z]?\)|Working Paper|Occasional Paper|Discussion Paper|"
    r"Macroprudential Bulletin|BIS central bankers|OJ L \d+|"
    r"Regulation \((EU|EC)\)|Directive \d+/\d+/(EU|EC)|ECB/\d{4}/\d+|"
    r"CON/\d{4}/\d+|pp?\.\s*\d+|^See\b|^See also\b|available at:|"
    r"eur-lex\.europa\.eu|doceo/document")

# "Surname, X.," author-initial lists — hallmark of reference entries
AUTHOR_SIG_RE = re.compile(
    r"[A-Z][A-Za-zà-ÿ'’\-]+,\s+[A-Z]\.(?=[,\s)]|$)"
    r"|\bby\s+(?:[A-Z]\.\s*)+[A-Z][a-zà-ÿ]+")
# quoted publication title followed by a (month) year — ECB pubs lists
PUBTITLE_RE = re.compile(
    rf"[“\"][^”\"]{{6,}}[”\"]\s*,\s*(?:({MONTHS})\s+)?\d{{4}}")
# legal-instrument pointers — hallmark of legal footnotes/endnotes
LEGAL_RE = re.compile(
    r"Regulation \((EU|EC)\)|Directive \d+/\d+|OJ L \d+|"
    r"Article \d+\(?\d*\)?|Decision \((EU|EC)\)|ECB/\d{4}/\d+")

FOOTNOTE_PARA_RE = re.compile(r"^\[?\d{1,2}\]?\s")

# --- content-start anchors that terminate a TOC zone ------------------------ #
TOC_END_ANCHOR_RE = re.compile(
    r"^(Foreword|FOREWORD|Preface|PREFACE|Editorial|EDITORIAL|Overview|"
    r"OVERVIEW|Executive summary|EXECUTIVE SUMMARY|Introduction|"
    r"INTRODUCTION|Summary|SUMMARY)$")
TOC_START_RE = re.compile(r"^(Contents|CONTENTS|Table of [Cc]ontents)$")

# --- sentence-level scrubs (applied inside kept paragraphs) ----------------- #
SENTENCE_SCRUB_RE = [
    re.compile(r"\s*The Opinion (?:was published in the Official Journal of "
               r"the EU on [^.]{3,40}?\s*(?:and|,)\s*)?is (?:also )?available "
               r"on the ECB.s website\.?"),
]

DISCLAIMER_PARA_RE = re.compile(r"^Disclaimer\s*:", re.I)

# --------------------------------------------------------------------------- #
#  Stage 1 — normalisation                                                    #
# --------------------------------------------------------------------------- #

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
             "ﬅ": "st", "ﬆ": "st"}


def normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # PDF ligatures often extract with a trailing space: "ﬁ nancial"
    for lig, plain in LIGATURES.items():
        text = text.replace(lig + " ", plain).replace(lig, plain)
    text = text.replace("\u00ad", "")            # soft hyphen
    text = text.replace("\u00a0", " ")           # nbsp
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)  # zero-width
    text = unicodedata.normalize("NFC", text)
    return text


# --------------------------------------------------------------------------- #
#  Line classifiers                                                           #
# --------------------------------------------------------------------------- #

NUMTOK_RE = re.compile(r"^[-+()\[\].,%*€$£¥/:\d\s]+$")
SHORT_LABEL_RE = re.compile(
    r"^(Q[1-4]|H[12]|[A-Z][a-z]{2}\.?|Total|Exports?|Imports?|Average|"
    r"average|Euro area enlargement|of which:?|memo item:?)\b.{0,30}$")


def is_numeric_line(line: str) -> bool:
    """Line that is table debris: mostly digits/punct, or short label."""
    s = line.strip()
    if not s:
        return False
    if NUMTOK_RE.match(s):
        return True
    alnum = [c for c in s if c.isalnum()]
    if alnum:
        digits = sum(c.isdigit() for c in alnum)
        if digits / len(alnum) > 0.45:
            return True
    toks = s.split()
    if len(toks) <= 4 and any(any(c.isdigit() for c in t) for t in toks) \
            and len(s) < 40:
        return True
    if SHORT_LABEL_RE.match(s) and len(toks) <= 5:
        return True
    return False


def is_letterspaced(line: str) -> bool:
    """Cover-art lines like 'A N N U A L  R E P O R T 1 9 9 9'."""
    toks = line.strip().split()
    if len(toks) < 4:
        return False
    single = sum(1 for t in toks if len(t) == 1)
    return single / len(toks) >= 0.6


def looks_prose(line: str, min_words: int = 12) -> bool:
    s = line.strip()
    if len(s.split()) < min_words:
        return False
    if s[:1] not in "\u201c\"" and not s[:1].isupper():
        return False
    return s[-1:] in ".?!\u201d\u2019\""


# --------------------------------------------------------------------------- #
#  Zone strips                                                                #
# --------------------------------------------------------------------------- #

def strip_head_nav(lines, log):
    """Remove leading website navigation tokens until first substantive line."""
    i = 0
    limit = min(len(lines), 60)
    while i < limit:
        s = lines[i].strip()
        if not s or s.lower() in NAV_TOKENS or s.rstrip(":").lower() in NAV_TOKENS:
            i += 1
            continue
        break
    if i:
        log["head_nav_lines"] = i
    return lines[i:]


def strip_tail(lines, log):
    """Cut from the first tail marker (chrome: last 30%; annexes: last 45%)."""
    n = len(lines)
    cut = None
    for i in range(int(n * 0.70), n):
        if any(rx.match(lines[i].strip()) for rx in TAIL_CUT_RE):
            cut = i
            break
    for i in range(int(n * 0.55), cut or n):
        if any(rx.match(lines[i].strip()) for rx in ANNEX_CUT_RE):
            cut = i
            break
    if cut is not None:
        log["tail_cut_lines"] = n - cut
        log["tail_cut_marker"] = lines[cut].strip()[:60]
        return lines[:cut]
    return lines


def strip_front_matter(lines, log):
    """PDF pubs open with 40+ lines of cover/imprint/TOC before the first
    real sentence. If the body starts late, keep only complete-sentence
    lines from that zone; web docs (short front matter) are untouched."""
    def block_is_prose(idx):
        """4 consecutive lines that together look like a wrapped paragraph."""
        chunk = [lines[k].strip() for k in range(idx, min(idx + 4, len(lines)))]
        if any(not c for c in chunk) or len(chunk) < 4:
            return False
        blob = " ".join(chunk)
        words = blob.split()
        digits = sum(c.isdigit() for c in blob)
        return (len(words) >= 30 and any(ch in blob for ch in ".?!")
                and digits / max(len(blob), 1) < 0.05
                and blob[:1].isupper())

    body = None
    for i, ln in enumerate(lines[:900]):
        s = ln.strip()
        if looks_prose(s, 14):
            body = i
            break
        # content-start heading (Preface/Foreword/...) not followed by a
        # bare TOC page number
        if TOC_END_ANCHOR_RE.match(s) and i > 30:
            k = i + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines) and not re.match(r"^\d{1,3}$", lines[k].strip()):
                body = i
                break
        if i > 30 and block_is_prose(i):
            body = i
            break
    if body is None or body <= 40:
        return lines
    kept, dropped = [], 0
    for ln in lines[:body]:
        if looks_prose(ln.strip(), 8):
            kept.append(ln)
        else:
            dropped += 1
    log["front_matter_lines"] = dropped
    return kept + [HARD_BREAK] + lines[body:]


def strip_toc(lines, log):
    """Remove Contents zone: entries end in page numbers / are short frags."""
    out, i, n = [], 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if TOC_START_RE.match(s) and i < n * 0.5:
            j, dropped, cap = i + 1, 0, 600
            while j < n and dropped < cap:
                t = lines[j].strip()
                if TOC_END_ANCHOR_RE.match(t):
                    # real heading only if not itself a TOC entry (page no. follows)
                    k = j + 1
                    while k < n and not lines[k].strip():
                        k += 1
                    if k >= n or not re.match(r"^\d{1,3}$", lines[k].strip()):
                        break
                if looks_prose(t, 12):
                    break
                # everything else (entries ending in page nos., box titles,
                # wrapped fragments, abbreviation glossaries) is still TOC
                j += 1
                dropped += 1
            log["toc_lines"] = log.get("toc_lines", 0) + (j - i)
            out.append(HARD_BREAK)
            i = j
            continue
        out.append(lines[i])
        i += 1
    return out


# --------------------------------------------------------------------------- #
#  Line & block filters                                                       #
# --------------------------------------------------------------------------- #

CAPS_HEAD_RE = re.compile(r"^[0-9IVX.\s]*[A-Z][A-Z0-9&,'’\-\s]+$")


def is_caps_heading(s: str) -> bool:
    if not CAPS_HEAD_RE.match(s):
        return False
    words = s.split()
    if not (2 <= len(words) <= 10):
        return False
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) >= 0.9


def filter_lines(lines, log):
    out, dropped = [], 0
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append("")
            continue
        if any(rx.match(s) for rx in LINE_KEEP_RE):
            out.append(ln)
            continue
        low = s.lower().rstrip(":")
        if (low in NAV_TOKENS
                or any(rx.match(s) for rx in LINE_DROP_RE)
                or is_letterspaced(s)
                or is_caps_heading(s)):
            out.append(HARD_BREAK)
            dropped += 1
            continue
        out.append(ln)
    log["line_filter_drops"] = dropped
    return out


def drop_runs(lines, pred, min_run, log, key, pad_short=True):
    """Drop maximal runs (>= min_run) of lines matching pred.
    Short/blank/label lines inside a run don't break it (pad_short)."""
    out, i, n, dropped = [], 0, len(lines), 0
    while i < n:
        if pred(lines[i]):
            j, hits = i, 0
            k = i
            while k < n:
                s = lines[k].strip()
                if pred(lines[k]):
                    hits += 1
                    j = k
                    k += 1
                elif pad_short and s and len(s.split()) <= 3 \
                        and any(c.isdigit() for c in s):
                    k += 1
                elif not s:
                    k += 1
                else:
                    break
            if hits >= min_run:
                dropped += (j - i + 1)
                out.append(HARD_BREAK)
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    if dropped:
        log[key] = log.get(key, 0) + dropped
    return out


def drop_numeric_tables(lines, log):
    return drop_runs(lines, is_numeric_line, 5, log, "table_lines")


def drop_single_char_runs(lines, log):
    return drop_runs(lines, lambda l: len(l.strip()) == 1, 4,
                     log, "cover_lines", pad_short=False)


def drop_rosters(lines, log):
    out, i, n, dropped = [], 0, len(lines), 0
    while i < n:
        s = lines[i].strip()
        if ROSTER_HEAD_RE.match(s):
            j, names = i + 1, 0
            while j < n:
                t = lines[j].strip()
                if not t or ROSTER_NAME_RE.match(t) or ROSTER_NOTE_RE.match(t) \
                        or ROSTER_HEAD_RE.match(t) or len(t.split()) <= 6:
                    if ROSTER_NAME_RE.match(t):
                        names += 1
                    j += 1
                    continue
                break
            if names >= 4:
                dropped += j - i
                out.append(HARD_BREAK)
                i = j
                continue
        out.append(lines[i])
        i += 1
    if dropped:
        log["roster_lines"] = dropped
    return out


def drop_references(lines, log, keep_footnotes=False):
    if keep_footnotes:
        return lines
    out, i, n, dropped = [], 0, len(lines), 0
    while i < n:
        if REFS_HEAD_RE.match(lines[i].strip()):
            j = i + 1
            while j < n:
                t = lines[j].strip()
                if t and looks_prose(t, 14) and not CITE_SIG_RE.search(t):
                    break
                j += 1
            dropped += j - i
            out.append(HARD_BREAK)
            i = j
            continue
        out.append(lines[i])
        i += 1
    if dropped:
        log["reference_lines"] = dropped
    return out


# --------------------------------------------------------------------------- #
#  De-hyphenation & reflow                                                    #
# --------------------------------------------------------------------------- #

WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")


def build_vocab(text: str):
    return set(w.lower() for w in WORD_RE.findall(text))


def dehyphenate(lines, vocab, log):
    out, i, n, joins = [], 0, len(lines), 0
    while i < n:
        ln = lines[i]
        m = re.search(r"([A-Za-zÀ-ÿ]{2,})-$", ln.rstrip())
        if m and i + 1 < n:
            nxt = lines[i + 1].lstrip()
            m2 = re.match(r"([a-zà-ÿ]{2,})(.*)$", nxt)
            if m2:
                w1, w2 = m.group(1), m2.group(1)
                merged = (w1 + w2).lower()
                hyphed = f"{w1.lower()}-{w2.lower()}"
                if merged in vocab or hyphed not in vocab:
                    joint = w1 + w2          # gather-ing -> gathering
                else:
                    joint = w1 + "-" + w2    # risk-neutral stays hyphenated
                lines[i + 1] = (ln.rstrip()[: -len(w1) - 1] + joint
                                + m2.group(2))
                joins += 1
                i += 1
                continue
        out.append(ln)
        i += 1
    log["hyphen_joins"] = joins
    return out


SENT_END = tuple(".?!:;\u201d\u2019\"")


def reflow(lines):
    """Join hard-wrapped lines into paragraphs.

    HARD_BREAK sentinels (filtered junk) and blank lines request a paragraph
    break — but a sentence sliced by a page break (running header removed
    mid-paragraph) is bridged: if the pending text does not end in sentence
    punctuation and the next line starts lowercase, the paragraph continues.
    """
    paras, cur, pending_break = [], [], False
    for ln in lines:
        if ln == HARD_BREAK or not ln.strip():
            pending_break = bool(cur)
            continue
        s = ln.strip()
        if not cur:
            cur, pending_break = [s], False
            continue
        if pending_break:
            joined = " ".join(cur)
            bridge = (joined[-1:] not in ".?!\u201d\u2019\""
                      and s[:1].islower()
                      and len(joined.split()) >= 6)
            if not bridge:
                paras.append(joined)
                cur = [s]
                pending_break = False
                continue
            pending_break = False
            cur.append(s)
            continue
        prev_end = cur[-1][-1:]
        starts_new = (prev_end in ".?!\u201d\"" and (s[:1].isupper()
                      or s[:1] in "\u201c\"“"))
        if starts_new and len(cur[-1].split()) >= 4:
            paras.append(" ".join(cur))
            cur = [s]
        else:
            cur.append(s)
    if cur:
        paras.append(" ".join(cur))
    return paras


# --------------------------------------------------------------------------- #
#  Paragraph scoring                                                          #
# --------------------------------------------------------------------------- #

def para_stats(p: str):
    toks = p.split()
    alpha = sum(c.isalpha() for c in p)
    digit = sum(c.isdigit() for c in p)
    return toks, alpha, digit


def keep_paragraph(p: str, min_words: int, keep_footnotes: bool) -> (bool, str):
    toks, alpha, digit = para_stats(p)
    n = len(toks)
    if n == 0:
        return False, "empty"
    if alpha and digit / max(alpha + digit, 1) > 0.35:
        return False, "numeric"
    if n < 50 and alpha and digit / max(alpha + digit, 1) > 0.22:
        return False, "numeric"          # chart-legend fragments
    if alpha / max(len(p), 1) < 0.55:
        return False, "low_alpha"
    if DISCLAIMER_PARA_RE.match(p):
        return False, "disclaimer"
    # abbreviation glossaries: dense runs of 2-3-letter ALL-CAPS tokens
    caps23 = sum(1 for t in toks if 2 <= len(t) <= 3 and t.isupper())
    if n >= 6 and caps23 / n >= 0.25:
        return False, "glossary"
    if not keep_footnotes:
        sigs = len(CITE_SIG_RE.findall(p))
        authors = len(AUTHOR_SIG_RE.findall(p))
        pubs = len(PUBTITLE_RE.findall(p))
        if authors >= 2 or (authors >= 1 and sigs >= 2) or sigs >= 3:
            return False, "citation"      # reference-list entries, any length
        quotes = len(re.findall(r"[\u201c\"][^\u201d\"]{6,}[\u201d\"]", p))
        years = len(re.findall(r"\b(?:19|20)\d{2}\b", p))
        if pubs >= 2 or (pubs >= 1 and n < 30) \
                or (quotes >= 2 and years >= 2) \
                or (n <= 25 and quotes >= 1 and years >= 1) \
                or (n <= 25 and years >= 1
                    and re.search(r"[\u201d\"] by [A-Z]", p)):
            return False, "pubs_list"     # ECB "documents published" annexes
        if sigs and (sigs / max(n / 12, 1) >= 1.0 or
                     (n < 60 and sigs >= 2) or
                     (FOOTNOTE_PARA_RE.match(p) and sigs >= 1)):
            return False, "citation"
        if n < 55 and LEGAL_RE.search(p):
            return False, "legal_footnote"
        if n < 45 and re.match(r"^(See|See also)\b", p):
            return False, "citation"
    if p[:1].islower() and n < 25:
        return False, "orphan"           # stranded wrap fragment
    if n <= 18 and re.search(r"\sS\d{1,3}$", p):
        return False, "annex_list"       # FSR chart-list entry + S-page ref
    # table column-header stacks: dangling hyphens / no sentence at all
    dangles = sum(1 for t in toks if len(t) > 2 and t.endswith("-"))
    if dangles >= 2:
        return False, "table_header"
    if (n >= 15 and not any(ch in p for ch in ".?!")
            and p[-1:] != ")"
            and not re.search(r"\b(19|20)\d{2}\b", p)
            and not re.match(r"^(Speech|Interview|Keynote|Statement|Remarks|"
                             r"Questions?|Introductory statement|Account of|"
                             r"Letter|Address)\b", p)):
        return False, "table_header"
    if n >= min_words:
        return True, ""
    # short paragraph rescue: real sentence / heading-like narrative
    if (p[:1].isupper() or p[:1] in "\u201c\"“") and p[-1:] in ".?!" and n >= 3:
        return True, ""
    if n >= 4 and p[:1].isupper() and not any(c.isdigit() for c in p):
        return True, ""          # section headings inside narrative
    return False, "stub"


# --------------------------------------------------------------------------- #
#  Post-fixes                                                                 #
# --------------------------------------------------------------------------- #

INLINE_FOOT_RE = re.compile(r"(?<=[a-z]\.)\d{1,2}(?=\s+[A-Z\u201c\"“])")
BRACKET_FOOT_RE = re.compile(r"\s*\[\s*\d{1,2}\s*\]")
ORDINAL_RE = re.compile(r"\b(\d+)\s+(st|nd|rd|th)\b")


def postfix(p: str) -> str:
    p = BRACKET_FOOT_RE.sub("", p)
    p = INLINE_FOOT_RE.sub("", p)
    p = ORDINAL_RE.sub(r"\1\2", p)
    for rx in SENTENCE_SCRUB_RE:
        p = rx.sub("", p)
    p = re.sub(r"\s+([,.;:?!%)])", r"\1", p)
    p = re.sub(r"([(“])\s+", r"\1", p)
    p = re.sub(r"\s{2,}", " ", p)
    p = re.sub(r"\s*Frankfurt am Main, (?:\d{1,2} )?\w+ \d{4}"
               r"(?:\s+[A-Z][\w.\u2019' ]{2,40})?$", "", p)
    return p.strip()


# --------------------------------------------------------------------------- #
#  Doc-type inference (from ECB filename conventions)                         #
# --------------------------------------------------------------------------- #

DOCTYPE_MAP = [
    (re.compile(r"is[-_]?stmt|introductory"), "introductory_statement"),
    (re.compile(r"ssm[-_]?in|(^|[-_])in([-_.]|$)"), "interview"),
    (re.compile(r"ssm[-_]?sp|(^|[-_])sp([-_.]|$)"), "speech"),
    (re.compile(r"(^|[-_])gc([-_.]|$)"), "governing_council_decisions"),
    (re.compile(r"account"), "monetary_policy_account"),
    (re.compile(r"(^|[-_])mb([-_.]|$)"), "monthly_bulletin"),
    (re.compile(r"ssm[-_]?ar"), "ssm_annual_report"),
    (re.compile(r"(^|[-_])ar([-_.]|$)"), "annual_report"),
    (re.compile(r"fsr"), "financial_stability_review"),
    (re.compile(r"(^|[-_])rb([-_.]|$)"), "research_bulletin"),
    (re.compile(r"mep"), "letter_to_mep"),
    (re.compile(r"mpb"), "macroprudential_bulletin"),
]


def infer_doctype(name: str) -> str:
    low = name.lower()
    for rx, label in DOCTYPE_MAP:
        if rx.search(low):
            return label
    return "unknown"


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #

def clean_document(raw: str, min_words: int, keep_footnotes: bool,
                   debug: bool = False):
    log = {}
    text = normalise(raw)
    lines = [l.rstrip() for l in text.split("\n")]

    lines = strip_head_nav(lines, log)
    lines = strip_tail(lines, log)
    lines = strip_front_matter(lines, log)
    lines = strip_toc(lines, log)
    lines = filter_lines(lines, log)
    lines = drop_single_char_runs(lines, log)
    lines = drop_rosters(lines, log)
    lines = drop_references(lines, log, keep_footnotes)
    lines = drop_numeric_tables(lines, log)

    vocab = build_vocab(text)
    lines = dehyphenate(lines, vocab, log)

    paras = reflow(lines)

    kept, drops = [], {}
    for p in paras:
        p2 = postfix(p)
        ok, why = keep_paragraph(p2, min_words, keep_footnotes)
        if ok and p2:
            kept.append(p2)
        else:
            drops[why] = drops.get(why, 0) + 1
            if debug and why not in ("empty", "stub"):
                print(f"    [drop:{why}] {p2[:90]!r}", file=sys.stderr)
    log["para_drops"] = drops
    log["paras_kept"] = len(kept)
    return "\n\n".join(kept), log


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--jsonl", help="also write a JSONL corpus file")
    ap.add_argument("--report", help="write a CSV cleaning report")
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--keep-footnotes", action="store_true",
                    help="retain citation/footnote paragraphs")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, jsonl_recs = [], []

    files = sorted(in_dir.rglob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {in_dir} (searched all subfolders)")

    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="replace")
        cleaned, log = clean_document(raw, args.min_words,
                                      args.keep_footnotes, args.debug)
        rel = fp.relative_to(in_dir)          # preserves subfolder structure
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(cleaned + "\n", encoding="utf-8")
        doctype = infer_doctype(fp.stem)
        pct = 100 * len(cleaned) / max(len(raw), 1)
        rows.append({
            "file": str(rel), "doc_type": doctype,
            "chars_in": len(raw), "chars_out": len(cleaned),
            "pct_kept": f"{pct:.1f}",
            "paras_kept": log.get("paras_kept", 0),
            "table_lines_dropped": log.get("table_lines", 0),
            "line_filter_drops": log.get("line_filter_drops", 0),
            "toc_lines": log.get("toc_lines", 0),
            "reference_lines": log.get("reference_lines", 0),
            "roster_lines": log.get("roster_lines", 0),
            "tail_cut_lines": log.get("tail_cut_lines", 0),
            "hyphen_joins": log.get("hyphen_joins", 0),
            "para_drops": json.dumps(log.get("para_drops", {})),
        })
        if args.jsonl:
            jsonl_recs.append({"source": str(rel), "doc_type": doctype,
                               "text": cleaned})
        print(f"  {str(rel):45s} {doctype:28s} "
              f"{len(raw):>8,} -> {len(cleaned):>8,} chars ({pct:5.1f}% kept)")

    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nReport  -> {args.report}")
    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as fh:
            for rec in jsonl_recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Corpus  -> {args.jsonl}")


if __name__ == "__main__":
    main()
