"""Annotation-prep pipeline for the scraped FOMC press-conference transcripts.

Sits next to fed_scrape.py / fed_tools.py / fed_sentences.py and reads the
text files that `python fed_scrape.py fomc-press-conferences` produced
(clean/fomc-press-conferences/{year}/*.txt, extracted from the official
transcript PDFs).  Press conferences exist from 2011-04-27 on (Bernanke's
first) -- quarterly through 2018, every meeting since 2019.  There were no
press conferences 1997-2010; the per-meeting record of Fed speech for those
years is the FOMC *meeting transcript* PDFs (released with a ~5-year lag),
and --in can be pointed at a scrape of those instead -- the cleaning here
is transcript-shaped for both formats.

For each file it:

  1. CLEANS down to the spoken body: drops the title block ("Transcript of
     Chair Powell's Press Conference"), the running page headers
     ("June 12, 2024 Chair Powell's Press Conference FINAL"), "Page N of M"
     footers, stage directions ("[Laughter]", "(inaudible)"), and mends
     end-of-line hyphenation from the PDF extraction.  Speaker labels
     ("CHAIR POWELL.", "MICHELLE SMITH.", reporters' names) are stripped
     from the text but kept as a `speaker` column; --chair-only keeps only
     the Chair's own turns (opening statement + answers, no questions).

  2. TOKENIZES into sentences with NLTK, as in Shah, Paturi & Chava (2023),
     "Trillion Dollar Words" -- imported from fed_sentences.py so the two
     corpora are tokenized identically.

  3. DROPS QUESTIONS: any sentence containing a question mark (reporters'
     questions, the moderator's prompts, the odd rhetorical one from the
     Chair) is omitted before filtering and splitting; --keep-questions
     retains them.  Then FILTERS with the paper's A1/B1 dictionary: target
     sentences only.  (No title filter here -- the paper applies that to
     speech files only; press conferences have no informative titles.)
     --no-target-filter keeps every non-question sentence instead.

  4. SPLITS target sentences exactly as in the paper (split_sentence,
     imported from fed_sentences.py): at "but", "however", "even though",
     "although", "while", ";", valid only if each segment contains a
     Table 1 word.

  5. SAMPLES 1,000 sentences (--n, seeded with --seed, default 42) with an
     EQUAL SHARE PER CALENDAR YEAR: n // n_years each, remainder spread
     over seeded-random years; a year with fewer sentences than its share
     contributes everything it has and the shortfall is re-spread over the
     years that still have spares.  The sample goes to
     presconf_to_annotate.xlsx with an empty `label` column (paper's
     scheme: 0 = Dovish, 1 = Hawkish, 2 = Neutral) and every sentence NOT
     selected goes to presconf_remaining.xlsx.  Both files carry each
     sentence's meeting date, source file, and speaker.

  6. DATE FILTER (--after): keeps only transcripts on or after a given
     date (inclusive), e.g. --after 2019-01-01.

Usage:
    pip install nltk pandas openpyxl
    python fed_scrape.py fomc-press-conferences      # scrape first
    python fed_presconf_sentences.py                 # -> out/*.xlsx
    python fed_presconf_sentences.py --n 1000 --seed 42
    python fed_presconf_sentences.py --chair-only    # drop reporters/moderator
    python fed_presconf_sentences.py --no-target-filter
    python fed_presconf_sentences.py --in clean/fomc-meeting-transcripts
"""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from fed_sentences import (TARGET_VOCAB, contains_any, split_sentence,
                           _sent_tokenize)

# ---------------------------------------------------------------------------
# Transcript furniture.  The FOMCpresconf PDFs carry a running head
# ("June 12, 2024 Chair Powell's Press Conference FINAL" -- sometimes
# extracted as one line, sometimes split across lines), a "Page N of M"
# footer, and a title block on page 1.  The meeting-transcript PDFs use
# "Meeting of the Federal Open Market Committee on ..." heads and bare
# "N of M" footers.  Conservative -- eyeball a few files per era and tune.

_FURNITURE_RES = [
    re.compile(r"^page \d{1,3} of \d{1,3}$", re.I),          # presconf footer
    re.compile(r"^\d{1,3} of \d{1,3}$"),                     # transcript footer
    re.compile(r"^(final|preliminary)$", re.I),              # head fragment
    re.compile(r"^[A-Z][a-z]+ \d{1,2}(?:[-\u2013]\d{1,2})?, \d{4}$"),  # bare date
    re.compile(r"^transcript of\b", re.I),                   # title line
    re.compile(r"^meeting of the federal open market committee\b", re.I),
    # Running head, with or without the leading date / trailing FINAL:
    # "June 12, 2024 Chair Powell's Press Conference FINAL".  [^.?!]*
    # keeps it from ever reaching across real sentence punctuation.
    re.compile(r"^(?:[A-Z][a-z]+ \d{1,2}, \d{4}[\s,]+)?"
               r"(?:chair(?:man|woman)?|vice chair(?:man)?)\b[^.?!]*"
               r"\bpress conference[\s,]*(?:final|preliminary)?$", re.I),
]

# Stage directions and inaudible markers, wherever they appear.
_STAGE_RE = re.compile(r"[\[(]\s*(?:laughter|applause|pause|inaudible|"
                       r"no response|off.?mic)[^\])]{0,40}[\])]", re.I)

# A speaker label at the start of a line: an honorific + surname
# ("CHAIR POWELL.", "MR. KOHN.") or a 2-4 word ALL-CAPS name
# ("MICHELLE SMITH.", "STEVE LIESMAN."), ending with a period.  Mixed-case
# labels (or a wrapped line that happens to start with an all-caps pair)
# will be mis-read -- rare in these PDFs, but eyeball the speaker column.
_SPEAKER_RE = re.compile(
    r"^(?P<sp>(?:CHAIR(?:MAN|WOMAN)?|VICE\s+CHAIR(?:MAN)?|GOVERNOR|PRESIDENT|"
    r"(?:MR|MS|MRS|DR)\.?)\s+[A-Z][A-Z.'\u2019-]*"
    r"|[A-Z][A-Z.'\u2019-]+(?:\s+[A-Z][A-Z.'\u2019-]+){1,3})"
    r"\.\s*(?P<rest>.*)$")

_LIGATURES = (("\ufb01", "fi"), ("\ufb02", "fl"), ("\u00ad", ""))


def parse_turns(text):
    """-> [(speaker, text)] speaking turns, furniture stripped.  Text before
    the first speaker label is title-page residue and is dropped -- unless
    no labels are found at all (unexpected layout), in which case the whole
    body is kept as one turn with speaker ''."""
    for lig, rep in _LIGATURES:
        text = text.replace(lig, rep)
    text = _STAGE_RE.sub(" ", text)
    text = re.sub(r"([a-z])-\n(?=[a-z])", r"\1", text)   # mend hyphenation
    turns, leading = [], []
    for raw in text.splitlines():
        ln = re.sub(r"\s+", " ", raw).strip()
        if not ln or any(rx.match(ln) for rx in _FURNITURE_RES):
            continue
        m = _SPEAKER_RE.match(ln)
        if m:
            sp = re.sub(r"\s+", " ", m.group("sp")).strip(" .")
            turns.append([sp, [m.group("rest").strip()]])
        elif turns:
            turns[-1][1].append(ln)
        else:
            leading.append(ln)
    if not turns:
        turns = [["", leading]]
    return [(sp, " ".join(l for l in lines if l)) for sp, lines in turns]


# ---------------------------------------------------------------------------

def build(in_dir, target_filter=True, do_split=True, chair_only=False,
          drop_questions=True, after=None):
    records = []
    n_files = n_dropped = n_turns = n_sents = n_quest = n_target = 0

    for path in sorted(Path(in_dir).rglob("*.txt")):
        m = re.match(r"(\d{4}-\d{2}-\d{2})", path.name)
        date = m.group(1) if m else ""
        if after and (not date or date < after):
            n_dropped += 1
            continue
        n_files += 1
        turns = parse_turns(path.read_text(encoding="utf-8", errors="replace"))
        if chair_only:
            turns = [t for t in turns if t[0].startswith("CHAIR")]
        for speaker, body in turns:
            n_turns += 1
            for sent in _sent_tokenize(body):
                n_sents += 1
                if drop_questions and "?" in sent:
                    n_quest += 1
                    continue            # questions (mostly reporters') omitted
                if target_filter and not contains_any(sent, TARGET_VOCAB):
                    continue            # paper: target sentences only (A1|B1)
                n_target += 1
                for unit in (split_sentence(sent) if do_split else [sent]):
                    records.append(dict(date=date, file=path.name,
                                        speaker=speaker, sentence=unit))

    msg = f"{n_files} transcripts"
    if after:
        msg += f", {n_dropped} dropped (before {after})"
    msg += (f"; {n_turns} speaking turns"
            + (" (Chair only)" if chair_only else "")
            + f", {n_sents} sentences"
            + (f" ({n_quest} questions dropped)" if drop_questions else "")
            + f", {n_target} target, {len(records)} after splitting")
    print(msg)
    return records


# ---------------------------------------------------------------------------

def stratified_sample(records, n, seed):
    """Equal share of n per calendar year, seeded.  Each year's quota is
    n // n_years, the remainder goes to seeded-random years; a year with
    fewer sentences than its quota contributes everything it has and the
    shortfall is re-spread over the years that still have spares."""
    rng = random.Random(seed)
    pools = defaultdict(list)
    for i, r in enumerate(records):
        pools[r["date"][:4]].append(i)
    years = sorted(pools)
    n = min(n, len(records))

    quota = dict.fromkeys(years, n // len(years))
    for y in rng.sample(years, n % len(years)):
        quota[y] += 1
    while True:                                   # re-spread shortfalls
        short = sum(quota[y] - len(pools[y]) for y in years
                    if quota[y] > len(pools[y]))
        for y in years:
            quota[y] = min(quota[y], len(pools[y]))
        if not short:
            break
        spare = [y for y in years if quota[y] < len(pools[y])]
        if not spare:
            break                                 # fewer than n sentences
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


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_dir", default="clean/fomc-press-conferences")
    ap.add_argument("--out", dest="out_dir", default="out")
    ap.add_argument("--n", type=int, default=1000,
                    help="sentences to sample for annotation (default 1000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chair-only", action="store_true",
                    help="keep only the Chair's own speaking turns "
                         "(no reporters, no moderator)")
    ap.add_argument("--keep-questions", action="store_true",
                    help="keep sentences containing '?' (dropped by default)")
    ap.add_argument("--no-target-filter", action="store_true",
                    help="keep every sentence, not only A1/B1 matches")
    ap.add_argument("--no-split", action="store_true",
                    help="skip the paper's sentence splitting")
    ap.add_argument("--after", default=None,
                    help="only transcripts on or after this date "
                         "(inclusive), e.g. 2019-01-01")
    args = ap.parse_args()

    if args.after and not re.match(r"^\d{4}-\d{2}-\d{2}$", args.after):
        raise SystemExit("--after must be in YYYY-MM-DD format, e.g. 2019-01-01")

    records = build(args.in_dir, target_filter=not args.no_target_filter,
                    do_split=not args.no_split, chair_only=args.chair_only,
                    drop_questions=not args.keep_questions, after=args.after)
    if not records:
        raise SystemExit(
            f"no sentences found under {args.in_dir} - run "
            "`python fed_scrape.py fomc-press-conferences` first"
            + (" (or drop --chair-only: no CHAIR-labelled turns found)"
               if args.chair_only else ""))

    ann, rest = stratified_sample(records, args.n, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["date", "file", "speaker", "sentence"]
    df_ann = pd.DataFrame(ann)[cols]
    df_ann["label"] = ""              # 0 = Dovish, 1 = Hawkish, 2 = Neutral
    df_rest = pd.DataFrame(rest)[cols] if rest else pd.DataFrame(columns=cols)
    p1 = out / "presconf_to_annotate.xlsx"
    p2 = out / "presconf_remaining.xlsx"
    df_ann.to_excel(p1, index=False)
    df_rest.to_excel(p2, index=False)
    print(f"wrote {len(df_ann)} sentences -> {p1}")
    print(f"wrote {len(df_rest)} sentences -> {p2}")


if __name__ == "__main__":
    main()
