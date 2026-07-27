"""Annotation-prep pipeline for the scraped Fed speeches.

Sits next to fed_scrape.py / fed_tools.py and reads the text files that
fed_scrape.py produced (clean/speeches/{year}/*.txt).  For each file it:

  1. CLEANS: strips everything that is not the speech itself -- the
     header block (date / title / speaker / venue), "Revised Version"
     notes, standalone footnote-number lines, "Return to text" links,
     section headings, and the entire tail (References, Footnotes,
     figure blocks, "Accessible Version").  The title and date are kept
     as metadata.

  2. TOKENIZES into sentences with NLTK (nltk.sent_tokenize), as in
     Shah, Paturi & Chava (2023), "Trillion Dollar Words".

  3. FILTERS with the paper's exact dictionary (their Table 1, taken
     from Gorodnichenko et al. 2021): a sentence is a "target" sentence
     iff it contains an instance of a word from panel A1 or panel B1.
     The same A1/B1 filter is applied to each speech TITLE first, as
     the paper does for speech files (201 of 1,026 files passed there).
     Disable with --no-title-filter to keep every file.

  4. SPLITS target sentences exactly as in the paper: at "but",
     "however", "even though", "although", "while", ";" -- a split is
     valid only if EACH segment contains a key word present in Table 1
     (the paper does not say which panels; this uses the whole table,
     panels A1+A2+B1+B2+C, and the vocabulary is one constant below if
     you want to narrow it).

  5. SAMPLES 1,000 sentences (--n, seeded with --seed) into an Excel
     file for annotation, with an empty `label` column (paper's scheme:
     0 = Dovish, 1 = Hawkish, 2 = Neutral), and writes ALL remaining
     sentences to a second Excel file.  Both files carry the date of
     every sentence's speech.  --per-file 5 instead mimics the paper's
     per-file sampling.

  6. DATE FILTER (--after): keeps only speeches on or after a given
     date, e.g. --after 2023-12-31 keeps everything from 2023-12-31
     onward (inclusive).  Filtering happens at the speech/file level
     (every sentence in a speech shares that speech's date), before
     the title filter and before sentences are extracted.

Matching is case-insensitive substring containment ("price" matches
"prices", "interest rate" matches "interest rates"), the literal
reading of the paper's "contained an instance of the words".

Usage:
    pip install nltk pandas openpyxl
    python fed_sentences.py                          # clean/speeches -> out/*.xlsx
    python fed_sentences.py --n 1000 --seed 42
    python fed_sentences.py --no-title-filter        # keep all files
    python fed_sentences.py --per-file 5             # paper-style sampling
    python fed_sentences.py --after 2023-12-31        # only speeches from that date on
"""

import argparse
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# The paper's Table 1 (Gorodnichenko et al. dictionary), verbatim.
PANEL_A1 = ["inflation expectation", "interest rate", "bank rate", "fund rate",
            "price", "economic activity", "inflation", "employment"]
PANEL_B1 = ["unemployment", "growth", "exchange rate", "productivity",
            "deficit", "demand", "job market", "monetary policy"]
PANEL_A2 = ["anchor", "cut", "subdue", "decline", "decrease", "reduce", "low",
            "drop", "fall", "fell", "decelerate", "slow", "pause", "pausing",
            "stable", "nonaccelerating", "downward", "tighten"]
PANEL_B2 = ["ease", "easing", "rise", "rising", "increase", "expand",
            "improve", "strong", "upward", "raise", "high", "rapid"]
PANEL_C = ["weren't", "were not", "wasn't", "was not", "did not", "didn't",
           "do not", "don't", "will not", "won't"]

TARGET_VOCAB = PANEL_A1 + PANEL_B1               # target-sentence + title filter
SPLIT_VALID_VOCAB = (PANEL_A1 + PANEL_A2         # split-validity check:
                     + PANEL_B1 + PANEL_B2       # "a key word present in
                     + PANEL_C)                  #  Table 1" -> whole table

# The paper's sentence-split keywords, verbatim.
SPLIT_RE = re.compile(r";|\b(?:even though|but|however|although|while)\b",
                      re.IGNORECASE)


def contains_any(text, vocab):
    t = text.lower()
    return any(w in t for w in vocab)


def split_sentence(sent):
    """Paper's scheme: split at the keywords; keep the split only if
    every segment contains a Table 1 word, else return the sentence
    unchanged."""
    parts = [p.strip(" ,\t") for p in SPLIT_RE.split(sent)]
    parts = [p for p in parts if p]
    if len(parts) >= 2 and all(contains_any(p, SPLIT_VALID_VOCAB) for p in parts):
        return parts
    return [sent]


# ---------------------------------------------------------------------------
# Date handling for --after.

def _normalize_date(date_str):
    """Best-effort conversion of a speech's date string to 'YYYY-MM-DD'
    so it can be compared against --after. The filename-derived date is
    already ISO ('2025-01-14') and passes straight through; the
    fallback date_in_text (e.g. 'January 14, 2025') is parsed. Returns
    None if the date can't be determined."""
    if not date_str:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Cleaning: strip everything that is not the speech.

_DATE_RE = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")
_SPEAKER_RE = re.compile(r"^(Governor|Chair|Chairman|Vice Chair|Vice Chairman|"
                         r"President|Member)\b")
# Lines that end the document: everything from the first of these on is
# references / footnotes / figures, not speech.
_TAIL_EXACT = {"References", "Footnotes", "Accessible Version"}
_TAIL_PREFIX = ("View speech charts", "Figure ")
_FOOTNOTE_START_RE = re.compile(r"^1\.\s+\S")   # "1. The views expressed..."
# Chrome lines dropped wherever they appear.
_JUNK_LINE_RE = re.compile(r"^(\d{1,2}|Return to text|Watch Live|Video|Share|"
                           r"PDF|Accessible Version|\* \* \*|\*\*\*)$")
_MOJIBAKE = (("\u00e2\u00e2", "--"), ("\u00e2", "'"), ("\u00c2", ""))


def _is_heading(line):
    """Section headings ('Conclusion', 'The Long Run', 'Monetary Policy'):
    short, no terminal punctuation.  Conservative -- tune if it bites."""
    return len(line.split()) <= 8 and not re.search(r"[.:,?!\"')\]]$", line)


def strip_non_speech(text):
    """-> (date_in_text, title, body).  Removes the header block, the
    References/Footnotes/figures tail, and chrome lines."""
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    lines = [ln.strip() for ln in text.splitlines()]

    # --- header: date, then title, then speaker/venue/revision lines ---
    date_in_text = title = None
    i, seen = 0, 0
    while i < len(lines) and seen < 12:
        ln = lines[i]
        if not ln:
            i += 1
            continue
        seen += 1
        if date_in_text is None and _DATE_RE.match(ln):
            date_in_text = ln
        elif title is None and not _SPEAKER_RE.match(ln):
            title = ln
        elif (_SPEAKER_RE.match(ln) or ln.startswith(("At ", "Before ", "(via",
                                                      "Revised Version"))
              or "original version of the remarks" in ln.lower()):
            pass                                  # speaker / venue / revision
        else:
            break                                 # first body line
        i += 1
    body_lines = lines[i:]

    # --- tail: cut at References / Footnotes / figures ---
    n = len(body_lines)
    cut = n
    for j, ln in enumerate(body_lines):
        if (ln in _TAIL_EXACT or ln.startswith(_TAIL_PREFIX)
                or (_FOOTNOTE_START_RE.match(ln) and j > 0.5 * n)):
            cut = j
            break
    body_lines = body_lines[:cut]

    # --- chrome + headings ---
    kept = [ln for ln in body_lines
            if ln and not _JUNK_LINE_RE.match(ln) and not _is_heading(ln)]
    return date_in_text, title, "\n".join(kept)


# ---------------------------------------------------------------------------
# Sentence tokenization (NLTK, as in the paper).

def _sent_tokenize(text):
    try:
        import nltk
        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:                       # noqa: BLE001
                    pass
        return nltk.sent_tokenize(text.replace("\n", " "))
    except (ImportError, LookupError):
        print("  ! NLTK punkt unavailable (not installed / offline?) - "
              "falling back to a regex tokenizer; `pip install nltk` for "
              "the paper's tokenization", file=sys.stderr)
        return [s.strip() for s in
                re.split(r"(?<=[.?!])\s+(?=[A-Z\"'(])", text.replace("\n", " "))
                if s.strip()]


# ---------------------------------------------------------------------------

def build(in_dir, title_filter=True, do_split=True, after=None):
    records = []
    n_files = n_kept = n_sents = n_target = 0
    n_date_dropped = n_undated_dropped = 0

    for path in sorted(Path(in_dir).rglob("*.txt")):
        n_files += 1
        m = re.match(r"(\d{4}-\d{2}-\d{2})", path.name)
        text = path.read_text(encoding="utf-8", errors="replace")
        date_in_text, title, body = strip_non_speech(text)
        date = m.group(1) if m else (date_in_text or "")

        # --- date filter: keep only speeches strictly after `after` ---
        if after:
            norm_date = _normalize_date(date)
            if norm_date is None:
                n_undated_dropped += 1
                continue
            if norm_date < after:
                n_date_dropped += 1
                continue

        if title_filter and title and not contains_any(title, TARGET_VOCAB):
            continue                    # paper: keep only A1/B1-matching titles
        n_kept += 1
        for sent in _sent_tokenize(body):
            n_sents += 1
            if not contains_any(sent, TARGET_VOCAB):
                continue                # paper: target sentences only (A1|B1)
            n_target += 1
            for unit in (split_sentence(sent) if do_split else [sent]):
                records.append(dict(date=date, file=path.name,
                                    title=title or "", sentence=unit))

    msg = (f"{n_files} files scanned")
    if after:
        msg += (f", {n_date_dropped} dropped (on/before {after}), "
                f"{n_undated_dropped} dropped (no parseable date)")
    msg += (f", {n_kept} kept"
            + (" after the title filter" if title_filter else "")
            + f"; {n_sents} sentences, {n_target} target, "
              f"{len(records)} after splitting")
    print(msg)
    return records


def sample_split(records, n, seed, per_file=None):
    rng = random.Random(seed)
    idx = list(range(len(records)))
    if per_file:
        chosen = []
        byfile = {}
        for i, r in enumerate(records):
            byfile.setdefault(r["file"], []).append(i)
        for ids in byfile.values():
            chosen += ids if len(ids) <= per_file else rng.sample(ids, per_file)
        chosen = set(chosen)
    else:
        chosen = set(idx if len(idx) <= n else rng.sample(idx, n))
    ann = [records[i] for i in idx if i in chosen]
    rest = [records[i] for i in idx if i not in chosen]
    return ann, rest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_dir", default="clean/speeches")
    ap.add_argument("--out", dest="out_dir", default="out")
    ap.add_argument("--n", type=int, default=1000,
                    help="sentences to sample for annotation (default 1000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-file", type=int, default=None,
                    help="sample up to N sentences per file instead "
                         "(the paper used 5)")
    ap.add_argument("--no-title-filter", action="store_true",
                    help="keep every file, not only A1/B1-matching titles")
    ap.add_argument("--no-split", action="store_true",
                    help="skip the paper's sentence splitting")
    ap.add_argument("--after", default=None,
                    help="only keep sentences from speeches on or after "
                         "this date (inclusive), e.g. 2023-12-31")
    args = ap.parse_args()

    if args.after and not re.match(r"^\d{4}-\d{2}-\d{2}$", args.after):
        raise SystemExit("--after must be in YYYY-MM-DD format, e.g. 2024-12-31")

    records = build(args.in_dir, title_filter=not args.no_title_filter,
                    do_split=not args.no_split, after=args.after)
    if not records:
        raise SystemExit(f"no sentences found under {args.in_dir} - "
                         "run `python fed_scrape.py speeches` first"
                         + (f" (or loosen --after {args.after})" if args.after else ""))

    ann, rest = sample_split(records, args.n, args.seed, args.per_file)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["date", "file", "title", "sentence"]
    df_ann = pd.DataFrame(ann)[cols]
    df_ann["label"] = ""              # 0 = Dovish, 1 = Hawkish, 2 = Neutral
    df_rest = pd.DataFrame(rest)[cols] if rest else pd.DataFrame(columns=cols)
    p1 = out / "speeches_to_annotate.xlsx"
    p2 = out / "speeches_remaining.xlsx"
    df_ann.to_excel(p1, index=False)
    df_rest.to_excel(p2, index=False)
    print(f"wrote {len(df_ann)} sentences -> {p1}")
    print(f"wrote {len(df_rest)} sentences -> {p2}")


if __name__ == "__main__":
    main()
