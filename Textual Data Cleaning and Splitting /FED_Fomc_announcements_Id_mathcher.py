# %% [markdown]
# # Standalone: rebuild sents (with dates) + match annotated IDs + attach dates
#
# Self-contained version -- runs top to bottom as a plain script OR as a
# notebook. It re-runs the full cleaning pipeline to reconstruct the complete
# `sents` frame (same rules as the original run, so indices 0..N-1 line up with
# the `id` column in your annotated holdout), then matches ids, attaches dates,
# prints a side-by-side verification, and saves the annotated file with dates.
#
# NOTE: the id matching is only valid if the cleaning rules here are IDENTICAL
# to the run that generated the holdout. The exact_match check at the end is
# precisely the test of that.

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
ANNOTATED_CSV = DATA_DIR / "fomc_statements_test_500.csv"          # labeled file (id + label)
OUTPUT_CSV    = DATA_DIR / "fomc_statements_test_500_with_dates.csv"
MATCH_REPORT  = DATA_DIR / "id_match_report.csv"

type_col = "Type"
date_col = None    # set manually (e.g. "Date") if auto-detection guesses wrong
text_col = None

# %% [markdown]
# ## Part 1 -- rebuild the full `sents` frame (pipeline, unchanged rules + date)

# %%
FOMC_unclean_data_csv = pd.read_csv(INPUT_CSV)

types = FOMC_unclean_data_csv[type_col].astype(str).str.strip()
is_statement = types.str.lower().str.contains(r"\bstatement\b", na=False)
is_minute = types.str.lower().str.contains(r"\bminute\b", na=False)
statements_df = FOMC_unclean_data_csv[is_statement].copy()
minutes_df = FOMC_unclean_data_csv[is_minute].copy()


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
        if len(sample) == 0:
            continue
        if pd.to_datetime(sample, errors="coerce").notna().mean() > 0.8:
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


# ---- normalisation --------------------------------------------------------

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
    r"^(?:\s*(?:"
    r"share|print|pdf"
    r"|for immediate release"
    r"|for release at[^\n]*?\d{4}"
    r"|last update[^\n]*?\d{4}"
    rf"|{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}"
    r"|\d{4}"
    r")\b[\s:,.\-]*)+",
    re.IGNORECASE,
)

def strip_leading_wrapper(s):
    return LEADING_WRAPPER.sub("", s).lstrip() if isinstance(s, str) else ""


# ---- original cleaning patterns ------------------------------------------

END_MARKERS = re.compile(
    r'\bvoting (?:for|against|in favor)\b'
    r'|\bvoted (?:for|against|to|in favor|unanimously)\b'
    r'|voting on (?:this|the) (?:action|matter)',
    re.I,
)
HEADER_MARKERS = re.compile(
    r'for immediate release|for release at|last update'
    r'|^\s*share\b|^\s*print\b|^\s*pdf\b',
    re.I,
)
NOISE = re.compile(
    r'for media inquiries|media inquiries|please (?:e-?mail|call)'
    r'|implementation note'
    r'|\bhttps?://|\bwww\.|federalreserve\.gov'
    r'|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    r'|[\w.+-]+@[\w-]+\.[\w.]+'
    r'|\[\s*email\s*protected\s*\]'
    r'|board of governors of the federal reserve',
    re.I,
)
ROSTER = re.compile(
    r'\b(?:chair(?:man)?|vice chair(?:man)?|governor|secretary|president)\b',
    re.I,
)


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


# ---- Table 1 dictionary ---------------------------------------------------

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

USE_PREFIX_MATCH = False


def _dict_regex(terms, prefix=USE_PREFIX_MATCH):
    parts = [r"\s+".join(re.escape(w) for w in t.split())
             for t in sorted(terms, key=len, reverse=True)]
    tail = r"[A-Za-z]*" if prefix else ""
    return re.compile(
        r"(?<![A-Za-z])(?:" + "|".join(parts) + r")" + tail + r"(?![A-Za-z])",
        re.IGNORECASE,
    )


TARGET_RE = _dict_regex(PANEL_A1 + PANEL_B1)
TABLE1_RE = _dict_regex(PANEL_A1 + PANEL_B1 + PANEL_A2 + PANEL_B2 + PANEL_C)


def is_target_sentence(s: str) -> bool:
    return bool(TARGET_RE.search(s)) if isinstance(s, str) else False


SPLIT_RE = re.compile(
    r"\s*(?:;|(?<![A-Za-z])(?:but|however|even\s+though|although|while)(?![A-Za-z]))\s*",
    re.IGNORECASE,
)


def split_on_connectives(sentence: str):
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


# ---- run the pipeline -----------------------------------------------------

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

before = len(sents)
sents = sents[~sents["sentence"].map(is_boilerplate_sentence)].reset_index(drop=True)
print(f"boilerplate guard removed: {before - len(sents)}")

before = len(sents)
sents = sents[sents["sentence"].map(is_target_sentence)].reset_index(drop=True)
print(f"filtration kept {len(sents)} / {before} sentences (Panel A1/B1 terms)")

n_pre = len(sents)
sents["sentence"] = sents["sentence"].map(split_on_connectives)
sents = sents.explode("sentence", ignore_index=True)
sents["sentence"] = sents["sentence"].astype(str).str.strip()
sents = sents[sents["sentence"].astype(bool)].reset_index(drop=True)
print(f"splitting: {n_pre} target sentences -> {len(sents)} segments")
print(f"FULL sents frame rebuilt: {len(sents)} rows (index 0..{len(sents)-1})")

# %% [markdown]
# ## Part 2 -- load annotated holdout + match ids + attach dates

# %%
annotated = pd.read_csv(ANNOTATED_CSV)
print(f"\nAnnotated file: {len(annotated)} rows, columns = {list(annotated.columns)}")

if "id" not in annotated.columns:
    raise KeyError("Annotated CSV has no 'id' column -- cannot match back.")

dup = annotated["id"].duplicated().sum()
if dup:
    print(f"WARNING: {dup} duplicate id(s) in the annotated file.")

n = len(sents)
in_range = annotated["id"].between(0, n - 1)
if not in_range.all():
    bad = annotated.loc[~in_range, "id"].tolist()
    print(f"WARNING: {len(bad)} id(s) out of range (0..{n-1}): "
          f"{bad[:10]}{' ...' if len(bad) > 10 else ''}")
    print("These get NaN dates -- likely the pipeline changed since the holdout "
          "was drawn.")

main_lookup = sents.reindex(annotated["id"])
annotated["date"] = main_lookup["date"].to_numpy()
main_sentences    = main_lookup["sentence"].to_numpy()
print(f"Matched dates for {annotated['date'].notna().sum()} / {len(annotated)} rows.")

# %% [markdown]
# ## Part 3 -- side-by-side verification (main sentence vs annotated sentence)

# %%
def _norm(s):
    return " ".join(str(s).split()) if pd.notna(s) else ""

comparison = pd.DataFrame({
    "id":                 annotated["id"].to_numpy(),
    "date":               annotated["date"].to_numpy(),
    "sentence_main":      main_sentences,
    "sentence_annotated": annotated["sentence"].to_numpy(),
    "label":              annotated["label"].to_numpy() if "label" in annotated.columns else "",
})
comparison["exact_match"] = [
    _norm(a) == _norm(b) and _norm(a) != ""
    for a, b in zip(comparison["sentence_main"], comparison["sentence_annotated"])
]

n_match = int(comparison["exact_match"].sum())
print(f"\nExact matches: {n_match} / {len(comparison)}")

# print a readable sample of the side-by-side (full table goes to CSV below)
print("\n--- Side-by-side sample (first 10) " + "-" * 40)
print(comparison[["id", "date", "sentence_main", "sentence_annotated", "exact_match"]]
      .head(10).to_string(index=False))

mismatches = comparison[~comparison["exact_match"]]
if len(mismatches) == 0:
    print("\nNo mismatches -- every annotated id points at the identical sentence. "
          "Dates are safe to use.")
else:
    print(f"\n{len(mismatches)} MISMATCH(ES) -- do NOT trust these dates until "
          f"you work out why the ids no longer line up:")
    print(mismatches[["id", "date", "sentence_main", "sentence_annotated"]]
          .to_string(index=False))

# %% [markdown]
# ## Part 4 -- save (annotated + dates to a NEW file; labels untouched)

# %%
if OUTPUT_CSV.exists():
    raise FileExistsError(f"{OUTPUT_CSV} already exists -- delete/rename it to regenerate.")

lead = [c for c in ["id", "date"] if c in annotated.columns]
rest = [c for c in annotated.columns if c not in lead]
annotated[lead + rest].to_csv(OUTPUT_CSV, index=False)
comparison.to_csv(MATCH_REPORT, index=False)

print(f"\nSaved annotated file WITH dates -> {OUTPUT_CSV.name}")
print(f"Saved full side-by-side report  -> {MATCH_REPORT.name}")