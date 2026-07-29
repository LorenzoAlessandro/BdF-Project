"""
Annotation-prep pipeline for the scraped ECB speeches.
Complete self-contained version for notebook use.
"""

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import nltk

# Download NLTK data if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# ---------------------------------------------------------------------------
# Vocabulary, tokenizer, and splitter (faithful re-implementation)

PANEL_A1 = ("inflation expectation", "interest rate", "bank rate",
            "fund rate", "price", "economic activity", "inflation",
            "employment")
PANEL_B1 = ("unemployment", "growth", "exchange rate", "productivity",
            "deficit", "demand", "job market", "monetary policy")
PANEL_A2 = ("anchor", "cut", "subdue", "decline", "decrease", "reduce",
            "low", "drop", "fall", "fell", "decelerate", "slow", "pause",
            "pausing", "stable", "non-accelerating", "downward", "tighten")
PANEL_B2 = ("ease", "easing", "rise", "rising", "increase", "expand",
            "improve", "strong", "upward", "raise", "high", "rapid")
PANEL_C = ("weren't", "were not", "wasn't", "was not", "did not",
           "didn't", "do not", "don't", "will not", "won't")

TARGET_VOCAB = PANEL_A1 + PANEL_B1              # target filter: A1 | B1
SPLIT_VOCAB = TARGET_VOCAB + PANEL_A2 + PANEL_B2 + PANEL_C  # any Table 1

_VOCAB_RES = {}

def _vocab_re(vocab):
    rx = _VOCAB_RES.get(vocab)
    if rx is None:
        alts = [re.escape(w).replace("\\ ", " ").replace(" ", r"\s+")
                for w in vocab]
        rx = re.compile(r"\b(?:" + "|".join(alts) + r")(?:e?s)?\b", re.I)
        _VOCAB_RES[vocab] = rx
    return rx

def contains_any(sentence, vocab):
    """True if the sentence contains any vocab word/phrase (whole-word,
    case-insensitive, optional plural 's'/'es')."""
    return bool(_vocab_re(tuple(vocab)).search(sentence))

_SPLIT_AT = re.compile(
    r";|\b(?:but|however|even\s+though|although|while)\b", re.I)

def split_sentence(sentence):
    """Paper's splitting scheme: split at the contrast keywords; valid
    only if EVERY segment contains a Table 1 word, else no split."""
    parts = [p.strip(" ,") for p in _SPLIT_AT.split(sentence)]
    parts = [p for p in parts if p]
    if len(parts) > 1 and all(contains_any(p, SPLIT_VOCAB) for p in parts):
        return parts
    return [sentence]

def _sent_tokenize(text):
    try:
        return nltk.tokenize.sent_tokenize(text)
    except LookupError:
        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        return nltk.tokenize.sent_tokenize(text)

# ---------------------------------------------------------------------------
# ECB Presidents (for final filtering)

ECB_PRESIDENTS = {
    # Willem F. Duisenberg (1998-2003)
    "willem f duisenberg", "w f duisenberg", "w.f. duisenberg", 
    "willem duisenberg", "duisenberg",
    # Jean-Claude Trichet (2003-2011)
    "jean-claude trichet", "jean claude trichet", "trichet",
    # Mario Draghi (2011-2019)
    "mario draghi", "draghi",
    # Christine Lagarde (2019-present)
    "christine lagarde", "lagarde"
}

def is_president_speaker(speaker):
    """Check if the speaker is an ECB President."""
    if not speaker:
        return False
    # Normalize the speaker name
    speaker_lower = speaker.lower().strip()
    # Remove titles like "President", "Mr", "Dr", etc.
    speaker_lower = re.sub(r'\b(president|mr|ms|mrs|dr|prof(?:essor)?)\.?\s+', '', speaker_lower)
    # Remove "by" at the start
    speaker_lower = re.sub(r'^by\s+', '', speaker_lower)
    # Normalize spaces and punctuation
    speaker_lower = re.sub(r'[.,\s]+', ' ', speaker_lower).strip()
    
    # Check against known President names
    for president in ECB_PRESIDENTS:
        if president in speaker_lower:
            return True
    return False

# ---------------------------------------------------------------------------
# ECB page furniture

_NORMALISE = (("\u00a0", " "), ("\u2019", "'"), ("\u2018", "'"),
              ("\u00ad", ""), ("\ufb01", "fi"), ("\ufb02", "fl"))

_FOOTNOTE_MARK = re.compile(r"\[\s*\d{1,3}\s*\]")   # "[1]", also split "[\n1\n]"

# First standalone occurrence of one of these lines ends the body.
_TAIL_LINES = {"contact", "media contacts", "annexes", "annex",
               "related topics", "disclaimer", "see also", "notes",
               "notes to editors", "are you happy with this page?"}
_TAIL_STARTS = ("reproduction is permitted",)

_FURNITURE_RES = [
    re.compile(r"^[A-Z][A-Z0-9 &''./-]{2,60}$"),        # ALL-CAPS chip/head
    re.compile(r"^.{0,70}?, \d{1,2} [A-Z][a-z]+ \d{4}\.?$"),  # "City, 22 June 2026"
    re.compile(r"^\d{1,2} [A-Z][a-z]+ \d{4}$"),               # bare date
    re.compile(r"^[*\s]+$"),                            # "* * *" separators
    re.compile(r"^skip to\b.*$", re.I),                 # search-overlay chrome
    re.compile(r"^(?:yes no|menu|suggestions|sort by|search options|"
               r"image preview|relevance|date|english|home|navigation|"
               r"content|footer|anytime|past month|past year)$", re.I),
    re.compile(r"^what made you unhappy\?$", re.I),
    re.compile(r"^page not working\b.*", re.I),
    re.compile(r"^(?:european central bank|european monetary institute|"
               r"press (?:and information )?division|"
               r"directorate general communications)$", re.I),
    re.compile(r"^(?:sonnemannstrasse|kaiserstrasse|postfach)\b.*", re.I),
    re.compile(r"^603\d\d frankfurt am main.*$", re.I),
    re.compile(r"^\+?49[ \d]+$"),
    re.compile(r"^(?:tel\.?|fax)[.: ].*$", re.I),
    re.compile(r"^[\w.-]+@ecb\.europa\.eu$", re.I),
    re.compile(r"^home\s+media\s+explainers\b.*", re.I),
    re.compile(r"^copyright \d{4}, european central bank$", re.I),
]

# Citation / reference sentences (this is what removes the footnote block).
_JUNK_RES = [
    re.compile(r"https?://|www\.", re.I),
    re.compile(r"^[A-Z][\w'-]+,\s+(?:[A-Z]\.\s*)+"),     # "Lagarde, C. (...)"
    re.compile(r"\b(?:working paper|occasional paper|discussion paper|"
               r"journal of|nber|op\.? ?cit|ibid|mimeo|forthcoming|"
               r"vol\.? ?\d|no\.? ?\d+, |pp?\. ?\d)\b", re.I),
    re.compile(r"^(?:see also\b|see\b|cf\.|source[s]?:)", re.I),
    re.compile(r"\(\d{4}[a-z]?\)[,:]?\s*[\"'\u201c\u201d]"),  # (2024), "Title
    re.compile(r"\bet al\.", re.I),
    # Editorial summary blocks paraphrase the speaker in the third person
    re.compile(r"\b(?:says|said|explains|explained|emphasises|emphasised|"
               r"argues|argued|notes|noted|adds|added|describes|described)\b"
               r".{0,60}\bECB\b"),
]

def _is_junk(sent, speaker=""):
    if len(sent.split()) < 5:                     # fragments, "Thank you." etc.
        return True
    if speaker:                                   # third-person self-mention =
        surname = speaker.split()[-1]             # summary/citation, not speech
        if len(surname) > 2 and re.search(r"\b" + re.escape(surname) + r"\b",
                                          sent):
            return True
    return any(rx.search(sent) for rx in _JUNK_RES)

# Head block: subtitle carries type, speaker, role, occasion.
_SUB_INST = re.compile(
    r"by\b.*(?:ECB|European Central Bank|European Monetary Institute|"
    r"\bEMI\b|Eurosystem)")            # no trailing \b: glued "ECBat" etc.
_SUB_ROLE = re.compile(
    r"\b(?:President|Vice[- ]?President|Member of the (?:Executive|"
    r"Supervisory) Board|Governor|Chair)\b")
_SUB_BY = re.compile(r"^[^.!?]{0,120}?by\b")   # early "by"

def _is_subtitle(ln):
    return (bool(_SUB_BY.match(ln)) and not ln.endswith(".")
            and bool(_SUB_INST.search(ln))
            and ("," in ln or _SUB_ROLE.search(ln)))

_SPEAKER_RE = re.compile(
    r"by\s+(?:the\s+)?(?:(?:Dr|Mr|Ms|Mrs|Prof(?:essor)?)\.?\s+)*"
    r"((?:[A-Z][\w.'-]*\s+){0,4}[A-Z][\w.'-]+)")
_NAME_CUT = re.compile(
    r"\s+(?:Member|President|Vice|Chair|Governor|Executive)\b.*$")

def _speaker(ln):
    m = _SPEAKER_RE.search(ln)
    if not m:
        return ""
    name = _NAME_CUT.sub("", m.group(1)).strip(" ,.")
    return name if " " in name else ""

_NAME_ROLE_RE = re.compile(
    r"^([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3}),\s+(?:President|"
    r"Vice[- ]?President|Member of the (?:Executive|Supervisory) Board|"
    r"Chair(?:man|woman)?|Governor)\b.*(?:ECB|European Central Bank|"
    r"European Monetary Institute|\bEMI\b|Eurosystem)")
_ROLE_RE = re.compile(
    r"^(?:the\s+)?(?:president|vice[- ]?president|member of the (?:executive|"
    r"supervisory) board|member of the ecb'?s? (?:executive|supervisory) "
    r"board|chair(?:man|woman)?)\b.*\b(?:ecb|european central "
    r"bank|european monetary institute|eurosystem|supervisory board)\b\.?$",
    re.I)
_NAME_RE = re.compile(r"^(?:[A-Z][\w.'-]+ ){1,3}[A-Z][\w.'-]+$")
_HEAD_SCAN = 12
_TERMINAL = tuple(".!?:;\"'\u201d\u2019)")

_DANGLING = frozenset("""a an the to of in on at by for with and or but nor
    as its our their his her your my is are was were be been being has have
    had will would shall should can could may might must that which who whom
    whose this these those than then so such very more most also not no into
    onto upon from about against between during towards toward under over
    within without per both either neither we they it i you when while if
    because across through""".split())
_JOIN_PUNCT = ",.;:)]%\u201d\u2019"
_END_OK = tuple(".!?\u2026\"'\u201d\u2019)")

def _mend(lines):
    out = []
    for ln in lines:
        if out:
            prev = out[-1]
            if (ln[0] in _JOIN_PUNCT
                    or not ln.strip("\"'\u201c\u201d\u2018\u2019")):
                out[-1] = prev + ln
                continue
            words = prev.rstrip("\"'\u201d\u2019").split()
            last = words[-1].lower().strip(",") if words else ""
            if (ln[0].islower()
                    or (not prev.endswith(_END_OK)
                        and (last in _DANGLING
                             or prev.endswith((",", "-", "\u2013", "\u2014"))))):
                out[-1] = prev + " " + ln
                continue
        out.append(ln)
    return out

def parse_speech(text):
    """-> (title, speaker, body_lines) with ECB head/tail furniture removed."""
    for a, b in _NORMALISE:
        text = text.replace(a, b)
    text = _FOOTNOTE_MARK.sub("", text)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    for i, ln in enumerate(lines):
        low = ln.lower().rstrip(":")
        if i and (low in _TAIL_LINES or low.startswith(_TAIL_STARTS)):
            lines = lines[:i]
            break
    fallback = next((m.group(1) for ln in lines[:30]
                     for m in [_NAME_ROLE_RE.match(ln)]
                     if m), "")
    lines = [ln for ln in lines
             if not any(rx.match(ln) for rx in _FURNITURE_RES)]

    title = speaker = ""
    sub_i = next((i for i, ln in enumerate(lines[:_HEAD_SCAN])
                  if _is_subtitle(ln)), None)
    if sub_i is not None:
        speaker = _speaker(lines[sub_i])
        for j in range(sub_i - 1, -1, -1):
            ln = lines[j]
            if (_ROLE_RE.match(ln)
                    or any(rx.match(ln) for rx in _FURNITURE_RES)
                    or (_NAME_RE.match(ln) and j + 1 < len(lines)
                        and _ROLE_RE.match(lines[j + 1]))):
                continue
            title = ln
            break
        body = lines[sub_i + 1:]
    else:
        t_i = next((i for i, ln in enumerate(lines[:_HEAD_SCAN])
                    if not any(rx.match(ln) for rx in _FURNITURE_RES)
                    and not _ROLE_RE.match(ln)), 0)
        title = lines[t_i] if lines else ""
        body = lines[t_i + 1:]
    if not speaker:
        speaker = fallback
    return title, speaker, body

def body_sentences(body_lines):
    """Furniture-free sentences: mend emphasis-split fragments, drop headings
    / list items / captions (short lines with no terminal punctuation), then
    tokenize per paragraph line so sentences never glue across paragraphs."""
    sents = []
    for ln in _mend(body_lines):
        if any(rx.match(ln) for rx in _FURNITURE_RES):
            continue
        if len(ln.split()) <= 12 and not ln.endswith(_TERMINAL):
            continue
        sents.extend(_sent_tokenize(ln))
    return sents

# ---------------------------------------------------------------------------

def load_topics(manifest):
    """clean-file name -> ECB topic tags string, from the scraper manifest."""
    out = {}
    p = Path(manifest)
    if not p.exists():
        return out
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = Path(row.get("clean_path", "") or "").name
            if name:
                out[name] = row.get("topics", "") or ""
    return out

def build(in_dir, glob="*.txt", title_filter=True, target_filter=True,
          do_split=True, drop_questions=False, after=None, topic=None,
          topics_by_file=None, president_only=True):
    """Process speeches and extract sentences."""
    records = []
    n_files = n_after = n_title = n_topic = 0
    n_sents = n_quest = n_junk = n_target = 0
    n_non_president = 0

    for path in sorted(Path(in_dir).rglob(glob)):
        m = re.match(r"(\d{4}-\d{2}-\d{2})", path.name)
        date = m.group(1) if m else ""
        if after and (not date or date < after):
            n_after += 1
            continue
        topics = (topics_by_file or {}).get(path.name, "")
        if topic is not None and topic.strip().lower() not in [
                t.strip().lower() for t in topics.split(";")]:
            n_topic += 1
            continue
        title, speaker, body = parse_speech(
            path.read_text(encoding="utf-8", errors="replace"))
        
        # Skip non-President speeches if president_only is True
        if president_only and not is_president_speaker(speaker):
            n_non_president += 1
            continue
            
        if title_filter and not contains_any(title, TARGET_VOCAB):
            n_title += 1
            continue
        n_files += 1
        for sent in body_sentences(body):
            n_sents += 1
            if drop_questions and "?" in sent:
                n_quest += 1
                continue
            if _is_junk(sent, speaker):
                n_junk += 1
                continue
            if target_filter and not contains_any(sent, TARGET_VOCAB):
                continue
            n_target += 1
            for unit in (split_sentence(sent) if do_split else [sent]):
                records.append(dict(date=date, file=path.name, speaker=speaker,
                                    title=title, topics=topics, sentence=unit))

    msg = f"{n_files} files kept"
    drops = []
    if after:
        drops.append(f"{n_after} before {after}")
    if topic is not None:
        drops.append(f"{n_topic} without topic '{topic}'")
    if title_filter:
        drops.append(f"{n_title} title-filtered")
    if president_only:
        drops.append(f"{n_non_president} non-President speakers")
    if drops:
        msg += " (" + ", ".join(drops) + " dropped)"
    msg += (f"; {n_sents} sentences, {n_junk} junk"
            + (f", {n_quest} questions" if drop_questions else "")
            + f" dropped; {n_target} target, {len(records)} after splitting")
    print(msg)
    return records

# ---------------------------------------------------------------------------

def stratified_sample(records, n, seed):
    """Equal share of n per calendar year, seeded."""
    rng = random.Random(seed)
    pools = defaultdict(list)
    for i, r in enumerate(records):
        pools[r["date"][:4]].append(i)
    years = sorted(pools)
    n = min(n, len(records))

    quota = dict.fromkeys(years, n // len(years))
    for y in rng.sample(years, n % len(years)):
        quota[y] += 1
    while True:
        short = sum(quota[y] - len(pools[y]) for y in years
                    if quota[y] > len(pools[y]))
        for y in years:
            quota[y] = min(quota[y], len(pools[y]))
        if not short:
            break
        spare = [y for y in years if quota[y] < len(pools[y])]
        if not spare:
            break
        base, extra = divmod(short, len(spare))
        bump = dict.fromkeys(spare, base)
        for y in rng.sample(spare, extra):
            bump[y] += 1
        for y in spare:
            quota[y] += bump[y]

    chosen = set()
    for y in years:
        chosen.update(rng.sample(pools[y], quota[y]))
    ann = [records[i] for i in range(len(records)) if i in chosen]
    rest = [records[i] for i in range(len(records)) if i not in chosen]
    print("per-year sample (picked/available): "
          + ", ".join(f"{y or '????'}: {quota[y]}/{len(pools[y])}"
                      for y in years))
    return ann, rest

def per_file_sample(records, k, seed):
    """k random sentences per file."""
    rng = random.Random(seed)
    pools = defaultdict(list)
    for i, r in enumerate(records):
        pools[r["file"]].append(i)
    chosen = set()
    for f in sorted(pools):
        idx = pools[f]
        chosen.update(idx if len(idx) <= k else rng.sample(idx, k))
    ann = [records[i] for i in range(len(records)) if i in chosen]
    rest = [records[i] for i in range(len(records)) if i not in chosen]
    print(f"per-file sample: up to {k}/file across {len(pools)} files "
          f"-> {len(ann)} sentences")
    return ann, rest

# ---------------------------------------------------------------------------
# Main execution function for notebook use

def process_ecb_speeches(
    in_dir="clean/speeches",
    glob_pattern="*.txt",
    out_dir="out",
    n_samples=1000,
    seed=42,
    per_file=None,
    topic=None,
    manifest="out/csv/speeches_manifest.csv",
    no_title_filter=False,
    no_target_filter=False,
    no_split=False,
    drop_questions=False,
    after=None,
    all_speakers=False
):
    """
    Main function to process ECB speeches and create annotation dataset.
    
    Parameters:
    -----------
    in_dir : str
        Directory containing scraped speech files
    glob_pattern : str
        File pattern to match (default: "*.txt")
    out_dir : str
        Output directory for Excel files
    n_samples : int
        Number of sentences to sample (default: 1000)
    seed : int
        Random seed for reproducibility (default: 42)
    per_file : int, optional
        Sample per-file instead of per-year
    topic : str, optional
        Filter by ECB topic tag
    manifest : str
        Path to manifest CSV with topic tags
    no_title_filter : bool
        Skip title filtering (keep all files)
    no_target_filter : bool
        Skip target sentence filtering (keep all sentences)
    no_split : bool
        Skip sentence splitting
    drop_questions : bool
        Drop sentences with question marks
    after : str, optional
        Only include files on or after this date (YYYY-MM-DD)
    all_speakers : bool
        Include all speakers (default: only Presidents)
    
    Returns:
    --------
    tuple: (annotate_df, remaining_df)
    """
    
    if after and not re.match(r"^\d{4}-\d{2}-\d{2}$", after):
        raise ValueError("--after must be in YYYY-MM-DD format")
    
    print("Loading topics from manifest...")
    topics_by_file = load_topics(manifest)
    if topic is not None and not topics_by_file:
        raise ValueError(f"Topic '{topic}' given but no topics found in {manifest}")
    
    print("Processing speeches...")
    records = build(in_dir, glob=glob_pattern,
                    title_filter=not no_title_filter,
                    target_filter=not no_target_filter,
                    do_split=not no_split,
                    drop_questions=drop_questions,
                    after=after, topic=topic,
                    topics_by_file=topics_by_file,
                    president_only=not all_speakers)
    
    if not records:
        raise ValueError(f"No sentences found under {in_dir}")
    
    print(f"Sampling {len(records)} sentences...")
    if per_file:
        ann, rest = per_file_sample(records, per_file, seed)
    else:
        ann, rest = stratified_sample(records, n_samples, seed)
    
    # Create DataFrames
    cols = ["date", "file", "speaker", "title", "topics", "sentence"]
    df_ann = pd.DataFrame(ann)[cols]
    df_ann["label"] = ""  # 0 = Dovish, 1 = Hawkish, 2 = Neutral
    
    df_rest = pd.DataFrame(rest)[cols] if rest else pd.DataFrame(columns=cols)
    
    # Save to Excel
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    p1 = out_path / "speeches_to_annotate.xlsx"
    p2 = out_path / "speeches_remaining.xlsx"
    
    df_ann.to_excel(p1, index=False)
    df_rest.to_excel(p2, index=False)
    
    print(f"Saved {len(df_ann)} sentences for annotation -> {p1}")
    print(f"Saved {len(df_rest)} remaining sentences -> {p2}")
    
    return df_ann, df_rest

# ---------------------------------------------------------------------------
# Example usage in notebook

if __name__ == "__main__":
    # Default usage - only Presidents, all filters applied
    df_annotate, df_remaining = process_ecb_speeches()
    
    # Alternative: include all speakers
    # df_annotate, df_remaining = process_ecb_speeches(all_speakers=True)
    
    # Alternative: filter by topic and no title filter
    # df_annotate, df_remaining = process_ecb_speeches(
    #     topic="Monetary policy",
    #     no_title_filter=True
    # )
    
    # Alternative: per-file sampling (paper's original method)
    # df_annotate, df_remaining = process_ecb_speeches(
    #     per_file=5,
    #     all_speakers=True
    # )