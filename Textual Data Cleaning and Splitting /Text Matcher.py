# %% [markdown]
# # Pass 2: text-based exact matching (for the ids that failed)
#
# The id-based pass left mismatches -- the pipeline no longer reproduces the
# exact indices from the original run. This script ignores ids entirely and
# instead scans the full `sents` frame for a WORD-FOR-WORD match of each
# annotated sentence, then assigns the date from the matched row.
#
# Matching is exact on the normalised text (whitespace collapsed, case kept):
#   - unique match  -> date assigned, method = "text_exact"
#   - no match      -> date NaN, method = "unmatched"  (sentence text itself
#                      changed between runs -- listed at the end for review)
#   - multiple rows with the same text but DIFFERENT dates -> ambiguous, date
#     NaN, method = "ambiguous" (FOMC boilerplate recurs across meetings, so
#     guessing one date would be wrong). If all duplicate rows share ONE date,
#     it's safe and assigned with method = "text_exact_dup".
#
# Run order: this script rebuilds the full sents frame itself (self-contained),
# loads the annotated CSV, matches by text, and writes NEW output files.
# The original annotated CSV is READ-ONLY here, same as before.

# %%
import re
import unicodedata
import pandas as pd
from pathlib import Path

pd.set_option("display.max_colwidth", 120)

# %% [markdown]
# ## Config -- paths (change for your machine!!)

# %%
DATA_DIR = Path("/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Data Folders/"
                "FED BRANCH/POLICY DECISIONS/FOMC ANNOUNCMENTS")

INPUT_CSV     = DATA_DIR / "fomc_minutes_statements.csv"
ANNOTATED_CSV = DATA_DIR / "fomc_statements_test_500.csv"            # labeled file (READ-ONLY)
OUTPUT_CSV    = DATA_DIR / "fomc_statements_test_500_dates_textmatch.csv"
MATCH_REPORT  = DATA_DIR / "text_match_report.csv"

type_col = "Type"
date_col = None
text_col = None

# %% [markdown]
# ## Part 1 -- rebuild the full `sents` frame with dates (identical pipeline)

# %%
FOMC_unclean_data_csv = pd.read_csv(INPUT_CSV)

types = FOMC_unclean_data_csv[type_col].astype(str).str.strip()
statements_df = FOMC_unclean_data_csv[types.str.lower().str.contains(r"\bstatement\b", na=False)].copy()
minutes_df    = FOMC_unclean_data_csv[types.str.lower().str.contains(r"\bminute\b",    na=False)].copy()


def _detect_text_col(df, exclude=()):
    cand = [c for c in df.columns if c not in exclude
            and not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])]
    L = {c: df[c].dropna().astype(str).str.len().mean() for c in cand}
    return max(L, key=L.get) if L else None


def _detect_date_col(df, exclude=()):
    for c in df.columns:
        if c not in exclude and pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        if c not in exclude and re.search(r"date|release", str(c), re.I):
            return c
    for c in df.columns:
        if c in exclude or pd.api.types.is_numeric_dtype(df[c]):
            continue
        sample = df[c].dropna().astype(str).head(50)
        if len(sample) and pd.to_datetime(sample, errors="coerce").notna().mean() > 0.8:
            return c
    return None


if date_col is None:
    date_col = _detect_date_col(statements_df, exclude={type_col})
if text_col is None:
    text_col = _detect_text_col(statements_df, exclude={type_col, date_col})
if date_col is None:
    raise ValueError("Could not auto-detect a date column. Set date_col manually. "
                     f"Available: {list(FOMC_unclean_data_csv.columns)}")

docs = pd.concat(
    [statements_df.assign(doc_type="statement"),
     minutes_df.assign(doc_type="minutes")],
    ignore_index=True,
)
docs = docs[[text_col, date_col, "doc_type"]].rename(
    columns={text_col: "raw_text", date_col: "date"}
)
docs["date"] = pd.to_datetime(docs["date"], errors="coerce").dt.strftime("%Y-%m-%d")
print("text column:", text_col, "| date column:", date_col, "| documents:", docs.shape)


def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'")
           .replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
           .replace("\u00a0", " "))
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


_MONTHS = (r"(?:january|february|march|april|may|june|july|august|september"
           r"|october|november|december)")
LEADING_WRAPPER = re.compile(
    r"^(?:\s*(?:share|print|pdf|for immediate release"
    r"|for release at[^\n]*?\d{4}|last update[^\n]*?\d{4}"
    rf"|{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}|\d{{4}})\b[\s:,.\-]*)+",
    re.IGNORECASE,
)

def strip_leading_wrapper(s):
    return LEADING_WRAPPER.sub("", s).lstrip() if isinstance(s, str) else ""


END_MARKERS = re.compile(
    r'\bvoting (?:for|against|in favor)\b'
    r'|\bvoted (?:for|against|to|in favor|unanimously)\b'
    r'|voting on (?:this|the) (?:action|matter)', re.I)
HEADER_MARKERS = re.compile(
    r'for immediate release|for release at|last update'
    r'|^\s*share\b|^\s*print\b|^\s*pdf\b', re.I)
NOISE = re.compile(
    r'for media inquiries|media inquiries|please (?:e-?mail|call)'
    r'|implementation note|\bhttps?://|\bwww\.|federalreserve\.gov'
    r'|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|[\w.+-]+@[\w-]+\.[\w.]+'
    r'|\[\s*email\s*protected\s*\]'
    r'|board of governors of the federal reserve', re.I)
ROSTER = re.compile(
    r'\b(?:chair(?:man)?|vice chair(?:man)?|governor|secretary|president)\b', re.I)


def keep_statement_sentences(sentences):
    out = []
    for s in sentences:
        s = (s or "").strip()
        if not s:
            continue
        if END_MARKERS.search(s):
            break
        if HEADER_MARKERS.search(s):
            continue
        if NOISE.search(s):
            continue
        out.append(s)
    return out


def clean_doc_sentences(sentences):
    kept = keep_statement_sentences(sentences)
    merged = []
    for s in kept:
        if len(s.split()) < 6 and merged:
            merged[-1] = merged[-1].rstrip() + ' ' + s
        else:
            merged.append(s)
    return merged


def is_boilerplate_sentence(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return True
    if END_MARKERS.search(s) or HEADER_MARKERS.search(s) or NOISE.search(s):
        return True
    seps = s.count(',') + s.count(';')
    tokens = s.split()
    if seps >= 3 and tokens:
        cap_ratio = sum(t[0].isupper() for t in tokens) / len(tokens)
        if cap_ratio > 0.6 and ROSTER.search(s):
            return True
    return False


PANEL_A1 = ["inflation expectation", "interest rate", "bank rate", "fund rate",
            "price", "economic activity", "inflation", "employment"]
PANEL_B1 = ["unemployment", "growth", "exchange rate", "productivity",
            "deficit", "demand", "job market", "monetary policy"]
PANEL_A2 = ["anchor", "cut", "subdue", "decline", "decrease", "reduce", "low",
            "drop", "fall", "fell", "decelerate", "slow", "pause", "pausing",
            "stable", "non-accelerating", "downward", "tighten"]
PANEL_B2 = ["ease", "easing", "rise", "rising", "increase", "expand", "improve",
            "strong", "upward", "raise", "high", "rapid"]
PANEL_C  = ["weren't", "were not", "wasn't", "was not", "did not", "didn't",
            "do not", "don't", "will not", "won't"]


def _dict_regex(terms, prefix=False):
    parts = [r"\s+".join(re.escape(w) for w in t.split())
             for t in sorted(terms, key=len, reverse=True)]
    tail = r"[A-Za-z]*" if prefix else ""
    return re.compile(
        r"(?<![A-Za-z])(?:" + "|".join(parts) + r")" + tail + r"(?![A-Za-z])",
        re.IGNORECASE)


TARGET_RE = _dict_regex(PANEL_A1 + PANEL_B1)
TABLE1_RE = _dict_regex(PANEL_A1 + PANEL_B1 + PANEL_A2 + PANEL_B2 + PANEL_C)

SPLIT_RE = re.compile(
    r"\s*(?:;|(?<![A-Za-z])(?:but|however|even\s+though|although|while)(?![A-Za-z]))\s*",
    re.IGNORECASE)


def split_on_connectives(sentence):
    if not isinstance(sentence, str):
        return []
    segments = [seg.strip() for seg in SPLIT_RE.split(sentence) if seg and seg.strip()]
    if len(segments) <= 1:
        return [sentence.strip()]
    if all(TABLE1_RE.search(seg) for seg in segments):
        return segments
    return [sentence.strip()]


def split(text):
    return re.split(r"(?<=[.!?])\s+", text) if isinstance(text, str) else []


docs["clean_text"] = docs["raw_text"].map(clean_text).map(strip_leading_wrapper)
docs["sentences"] = docs["clean_text"].map(split).map(clean_doc_sentences)

sents = (
    docs.explode("sentences", ignore_index=True)
        .rename(columns={"sentences": "sentence"})
        .dropna(subset=["sentence"])
)
sents["sentence"] = sents["sentence"].astype(str).str.strip()
sents = sents[sents["sentence"].astype(bool)].reset_index(drop=True)
sents = sents[["date", "doc_type", "sentence"]]
sents = sents[~sents["sentence"].map(is_boilerplate_sentence)].reset_index(drop=True)
sents = sents[sents["sentence"].map(lambda s: bool(TARGET_RE.search(s)))].reset_index(drop=True)
sents["sentence"] = sents["sentence"].map(split_on_connectives)
sents = sents.explode("sentence", ignore_index=True)
sents["sentence"] = sents["sentence"].astype(str).str.strip()
sents = sents[sents["sentence"].astype(bool)].reset_index(drop=True)
print(f"FULL sents frame rebuilt: {len(sents)} rows")

# %% [markdown]
# ## Part 2 -- build the text lookup (normalised sentence -> set of dates)

# %%
def _norm(s):
    """Whitespace-collapsed exact form used for word-for-word matching."""
    return " ".join(str(s).split()) if pd.notna(s) else ""

sents["_key"] = sents["sentence"].map(_norm)

# sentence text -> unique dates it appears under (FOMC boilerplate recurs!)
key_to_dates = (
    sents.groupby("_key")["date"]
         .agg(lambda x: sorted(set(d for d in x if pd.notna(d))))
)

n_dup_text = (sents["_key"].duplicated(keep=False)).sum()
print(f"Note: {n_dup_text} rows in the main frame share their exact text with "
      f"at least one other row (recurring boilerplate across meetings).")

# %% [markdown]
# ## Part 3 -- load annotated file and match word-for-word

# %%
annotated = pd.read_csv(ANNOTATED_CSV)
print(f"Annotated file: {len(annotated)} rows")

results = []
for _, row in annotated.iterrows():
    key = _norm(row["sentence"])
    dates = key_to_dates.get(key, None)
    if not key or dates is None or len(dates) == 0:
        results.append({"date": pd.NA, "match_method": "unmatched",
                        "n_candidate_dates": 0, "candidate_dates": ""})
    elif len(dates) == 1:
        results.append({"date": dates[0], "match_method": "text_exact",
                        "n_candidate_dates": 1, "candidate_dates": dates[0]})
    else:
        # same sentence appears under several different dates -> cannot pick one
        results.append({"date": pd.NA, "match_method": "ambiguous",
                        "n_candidate_dates": len(dates),
                        "candidate_dates": "; ".join(dates)})

res_df = pd.DataFrame(results)
annotated_out = pd.concat([annotated.reset_index(drop=True), res_df], axis=1)

# %% [markdown]
# ## Part 4 -- report

# %%
counts = annotated_out["match_method"].value_counts()
print("\nMatch summary:")
for method, c in counts.items():
    print(f"  {method:>12}: {c}")

unmatched = annotated_out[annotated_out["match_method"] == "unmatched"]
if len(unmatched):
    print(f"\n--- UNMATCHED ({len(unmatched)}) -- sentence text not found "
          f"word-for-word in the rebuilt frame " + "-" * 20)
    print(unmatched[["id", "sentence"]].to_string(index=False))
    print("\nThese usually mean a cleaning rule changed the sentence text "
          "itself (e.g. different splitter, merged fragments). They need "
          "manual dating or a fuzzy pass.")

ambiguous = annotated_out[annotated_out["match_method"] == "ambiguous"]
if len(ambiguous):
    print(f"\n--- AMBIGUOUS ({len(ambiguous)}) -- exact text appears under "
          f"multiple dates " + "-" * 20)
    print(ambiguous[["id", "sentence", "candidate_dates"]].to_string(index=False))
    print("\nRecurring FOMC boilerplate: the same sentence was said at several "
          "meetings, so no single date can be assigned from text alone. "
          "candidate_dates lists all of them.")

# %% [markdown]
# ## Part 5 -- save (original annotated CSV is NOT touched)

# %%
if OUTPUT_CSV.exists():
    raise FileExistsError(f"{OUTPUT_CSV} already exists -- delete/rename it to regenerate.")

lead = [c for c in ["id", "date", "match_method"] if c in annotated_out.columns]
rest = [c for c in annotated_out.columns if c not in lead]
annotated_out[lead + rest].to_csv(OUTPUT_CSV, index=False)

annotated_out.to_csv(MATCH_REPORT, index=False)

n_dated = annotated_out["date"].notna().sum()
print(f"\nSaved -> {OUTPUT_CSV.name}")
print(f"Saved -> {MATCH_REPORT.name}")
print(f"Dates assigned: {n_dated} / {len(annotated_out)} "
      f"(unmatched: {len(unmatched)}, ambiguous: {len(ambiguous)})")