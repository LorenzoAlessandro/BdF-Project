# The following module contains my augmented cleaning function, which is designed to clean the boiler plate that can be found at the
# top and bottom of the majority of my scraped FED documents (Monetary Policy Reports, Financial Stability Reports, FOMC material
# etc.). Although most documents were cleaned during the scraping phase, there still could be remnants that could potentially pose
# issues, especially now that the cleaned text is destined for continued pre-training of an LLM, where any residual boiler plate,
# chart debris, or broken sentences would be learned directly by the model.
#
# Alongside boiler plate issues, most notably for the Monetary Policy and Financial Stability Reports, there are many graphs which
# are outside the scope of this project. Since in the initial scrape they were removed, yet their headers, axis labels, and
# footnotes may have been picked up, this therefore requires me to remove them from the text.
#
# This is a drop-in replacement for my previous clean_text(), it keeps the same name and the same drop_notes default, with a few
# optional knobs added on top. The order of operations inside it matters, the line-level junk removal now runs before the
# de-hyphenation, because a running header often sits between the two halves of a word hyphenated across a page break
# ("communi-\nBoard of Governors...\ncation") and it must be removed before the two halves can be re-joined, my previous version
# joined hyphens first, which meant those words could never be repaired. Blank lines (paragraph boundaries) are also preserved all
# the way through, since the paragraph reflow at the end depends on them to unwrap the hard line breaks that PDF extraction leaves
# every ~45-90 characters, so that the LLM trains on natural paragraphs instead of arbitrarily chopped lines.
#
# Corpus-level steps that belong OUTSIDE this function: exact/near-duplicate deduplication across documents (the same MPR often
# exists as both an HTML and a PDF scrape), dropping documents that come out shorter than some minimum after cleaning
# (min_alpha_chars), and manually spot checking a random sample of the cleaned outputs before launching the training run.

import re
import unicodedata
from collections import Counter

# I used the ftfy (fixes text for you) library to fix the standard/common Unicode encoding issues (mojibake such as "â€™" -> "'").
# It is strongly recommended for my corpus, however the pipeline degrades gracefully if it is not installed.
try:
    import ftfy
    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False

# Precomposed vulgar fractions are mapped BEFORE the NFKC normalization. NFKC would otherwise turn "5¼" into "51⁄4" (the superscript
# and subscript digits get flattened and glued onto the preceding integer), which is exactly the ambiguity that broke my previous
# fraction regex, that version actually produced "51 4" rather than "5 1/4", and it also rewrote dates ("1/2/2004" -> "1 2/2004").
_VULGAR_FRACTIONS = {
    "\u00bc": " 1/4", "\u00bd": " 1/2", "\u00be": " 3/4",
    "\u2153": " 1/3", "\u2154": " 2/3",
    "\u2155": " 1/5", "\u2156": " 2/5", "\u2157": " 3/5", "\u2158": " 4/5",
    "\u2159": " 1/6", "\u215a": " 5/6",
    "\u215b": " 1/8", "\u215c": " 3/8", "\u215d": " 5/8", "\u215e": " 7/8",
}

# Translation tables which map the Unicode superscript and subscript digits onto their standard ASCII digits.
_SUPERSCRIPT_DIGITS = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
_SUBSCRIPT_DIGITS = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
_SUP_TO_ASCII = str.maketrans(_SUPERSCRIPT_DIGITS, "0123456789")
_SUB_TO_ASCII = str.maketrans(_SUBSCRIPT_DIGITS, "0123456789")

# Reconstructs fractions typeset with superscript/subscript digits, "5¹⁄₄" -> "5 1/4". This runs before NFKC for the same reason as
# the vulgar fraction map above.
_RE_SUPSUB_FRACTION = re.compile(
    "([%s]+)\\s*\u2044\\s*([%s]+)" % (_SUPERSCRIPT_DIGITS, _SUBSCRIPT_DIGITS)
)

# Reconstructs fractions that were already glued together by an earlier NFKC pass in my scraper, "51⁄4 percent" -> "5 1/4 percent".
# This only fires on U+2044 FRACTION SLASH, never on the ordinary "/", so dates like 1/2/2004 are untouched. Single-digit numerators
# only, "5 11⁄16" stays ambiguous and is deliberately left alone.
_RE_GLUED_FRACTION = re.compile("(\\d+)([1-9])\u2044(\\d{1,2})")

# Fix soft hyphens and line-break hyphens, effectively similar to the standard convention which is seen in the PDF versions of the
# reports, many words at the end of a page contain hyphens, such that "communication" at the end of a page will read
# "communi- (new line) cation", this is removed and accounted for here (the visible "-\n" case is handled further down, after the
# boiler plate that sits between the two halves has been removed).
_RE_SOFT_HYPHEN_JOIN = re.compile("(\\w)\u00ad\\s*\\n?\\s*(\\w)")

# Normalizes hyphen look-alikes (Unicode hyphen, non-breaking hyphen, figure dash, minus sign) to the ASCII "-". En and em dashes
# are real punctuation and are kept as they are.
_RE_HYPHEN_VARIANTS = re.compile("[\u2010\u2011\u2012\u2212]")

# Replaces middle dots and bullet points with spaces, including the black squares used in the MPR running headers
# ("Monetary Policy Report ▪ February 2019").
_RE_BULLETS = re.compile("[\u00b7\u2022\u2023\u25aa\u25a0\u25cf\u25b6\u25b8\u2043]+")

# Removes zero-width spaces, the byte order mark, and the Unicode replacement character.
_RE_INVISIBLES = re.compile("[\u200b-\u200f\ufeff\ufffd]")

# Removes any C0 control characters other than newlines and tabs, these occasionally survive PDF extraction (form feeds especially).
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _normalize_unicode(text):
    # Unicode normalization is the initial step since I need to ensure that the text remains consistent and predictable throughout
    # the whole document before any of the pattern matching runs.
    if _HAS_FTFY:
        text = ftfy.fix_text(text)
    # Converts the Unicode line and paragraph separators into ordinary newlines so the line-level filtering sees them.
    text = text.replace("\u2028", "\n").replace("\u2029", "\n\n")
    # Reconstructs superscript/subscript fractions, "5¹⁄₄" -> "5 1/4", before NFKC can glue them together.
    text = _RE_SUPSUB_FRACTION.sub(
        lambda m: " %s/%s" % (m.group(1).translate(_SUP_TO_ASCII),
                              m.group(2).translate(_SUB_TO_ASCII)),
        text,
    )
    # Maps the precomposed vulgar fractions, "5¾" -> "5 3/4", again before NFKC for the reason explained above.
    for char, replacement in _VULGAR_FRACTIONS.items():
        text = text.replace(char, replacement)
    # Normalizes Unicode characters to NFKC, which stands for Normalization Form Compatibility Composition, this also flattens
    # ligatures (ﬁ -> fi) which appear frequently in the typeset PDFs.
    text = unicodedata.normalize("NFKC", text)
    # Reconstructs any fraction that was already glued in the source data, "51⁄4" -> "5 1/4" (U+2044 only, so dates are safe).
    text = _RE_GLUED_FRACTION.sub(r"\1 \2/\3", text)
    # Any remaining fraction slash becomes an ordinary slash.
    text = text.replace("\u2044", "/")
    # Replaces non-breaking spaces with regular spaces (NFKC already handles this, I keep it as belt and braces).
    text = text.replace("\u00a0", " ")
    # Removes zero-width spaces and the byte order mark.
    text = _RE_INVISIBLES.sub("", text)
    # Joins words that were split by a soft hyphen, with or without a line break.
    text = _RE_SOFT_HYPHEN_JOIN.sub(r"\1\2", text)
    # Removes any remaining soft hyphens.
    text = text.replace("\u00ad", "")
    # Normalizes the hyphen look-alikes to ASCII "-".
    text = _RE_HYPHEN_VARIANTS.sub("-", text)
    # Replaces middle dots and bullet points with spaces.
    text = _RE_BULLETS.sub(" ", text)
    # Strips residual control characters.
    text = _RE_CONTROL.sub(" ", text)
    return text


# REMOVE ALL BOILERPLATE - every pattern below is applied with fullmatch() against a single stripped line. This is the crucial
# difference from my previous version, where unanchored patterns such as "February \d{1,2}, \d{4}", "Wednesday", "Contents", and
# "Figure\s+\d+" (all with IGNORECASE) deleted words out of the middle of body sentences and left broken text behind. Full-line
# matching means that an in-sentence "On February 14, 2023, the Committee..." survives, while a standalone dateline on the cover
# page is still dropped.

_MONTHS = (r"(?:january|february|march|april|may|june|july|august|"
           r"september|october|november|december)")

# Case-insensitive full-line boiler plate.
_CI_LINE_PATTERNS = [
    # Cover page / transmittal letter furniture.
    r"for (?:use|release) at \d{1,2}:\d{2}\s*[ap]\.?\s?m\.?,?\s*e?[cdmps]*\.?[ds]?\.?t\.?\,?.*",
    r"embargoed for release.*",
    r"board of governors of the federal reserve system(?:\s*[|,].*)?",
    r"monetary policy report(?: to the congress)?(?:\s*[|,]?\s*" + _MONTHS + r"\s+\d{4})?",
    r"financial stability report(?:\s*[|,]?\s*" + _MONTHS + r"\s+\d{4})?",
    r"letter of transmittal",
    r"the president of the senate",
    r"the speaker of the house(?: of representatives)?",
    r"dear (?:mr|madam|ms|mrs)\.?\s+(?:president|speaker)\s*[,.:;]?",
    r"(?:submitted )?pursuant to section 2b of the federal reserve act.*",
    # Any chair's signature line, not just Greenspan's, since my corpus spans decades (Greenspan, Bernanke, Yellen, Powell).
    r"[a-z .,'-]{3,40},\s*chair(?:man|woman)?(?:\s+pro tempore)?",
    r"washington,?\s*d\.?\s?c\.?,?(?:\s+" + _MONTHS + r"\s+\d{1,2},?\s+\d{4})?\.?",
    r"sincerely,?",
    r"contents",
    r"list of (?:boxes|figures|tables)",
    r"(?:report )?submitted to the congress on\s+.*",
    r"page(?:\s+\d+)?",
    # Website chrome, in case some of my documents were scraped from federalreserve.gov HTML rather than the PDFs.
    r"skip to main content",
    r"back to top",
    r"last update:?\s*.*",
    r"accessible version(?:\s+of.*)?",
    r"return to text",
    r"printable version",
    r"pdf(?:\s*\|\s*html)?",
    r"https?://\S+",
    r"www\.\S+",
    # Standalone datelines / weekday lines, full line only, therefore in-sentence dates survive.
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day(?:,\s+" + _MONTHS + r"\s+\d{1,2},?\s+\d{4})?",
    _MONTHS + r"\s+\d{1,2},?\s+\d{4}",
    # Chart axis / unit lines.
    r"percent(?:age points)?(?:,\s*annual rate)?",
    r"percent,\s*(?:monthly|quarterly|weekly|daily|annual rate)",
    r"(?:billions?|trillions?|millions?|thousands?) of (?:\d{4} )?(?:chained\s*\(\d{4}\)\s*)?dollars",
    r"dollars per \w+",
    r"basis points",
    r"index(?:,.*=\s*100.*)?",
    r".{0,30}=\s*100",
    r"ratio(?:\s+scale)?",
    r"log scale.*",
    r"\d{1,2}-month percent change",
    r"annual(?:ized)? rate",
    r"seasonally adjusted(?:\s+annual rate)?",
    r"(?:quarterly|monthly|weekly|daily)(?:\s+average)?",
    r"panel\s+[a-z]",
    r"(?:\()?continued(?:\))?",
    r"h\.\d{1,2}(?:\.\d+)?(?:\s+release)?",   # Statistical release codes, e.g. H.15.
]
_CI_LINE = re.compile(
    "(?:%s)" % "|".join("(?:%s)" % p for p in _CI_LINE_PATTERNS), re.IGNORECASE
)

# Case-SENSITIVE full-line patterns. These are kept case-sensitive on purpose, a body sentence wrapped mid-line starts in lowercase
# ("...as shown in\nfigure 2, rates fell"), so requiring the capitalized caption form plus punctuation ("Figure 2. Change in real
# GDP") avoids eating the in-text cross-references which my previous version was destroying.
_CS_LINE_PATTERNS = [
    r"(?:Figure|Chart|Table|Box|Exhibit)\s+[A-D]?\.?\d+(?:\.\d+)?[.:]\s*\S.*",
    r"(?:Figure|Chart|Table|Box|Exhibit)\s+[A-D]?\.?\d+(?:\.\d+)?",
    r"[A-Z][a-z]+\s+[A-Z][a-z]+\s+\d+",       # Axis tick rows like "January February 2004".
    # Remove ALL chart titles and figure labels, following the same protocol as before. Using Artificial Intelligence Tools, I gave
    # extracts of each year of all reports and told the LLM to extract all relevant boiler plates and graph outlines. I have kept
    # that curated list, but it is now full-line + case-sensitive, so that a wrapped body line beginning "net percentage of domestic
    # banks tightening..." survives while the chart title itself is still removed.
    r"Selected interest rates",
    r"Ten-year Treasury",
    r"Two-year Treasury",
    r"Intended federal funds rate",
    r"Change in real GDP",
    r"Change in PCE chain-type price index",
    r"Change in real income and consumption",
    r"Wealth-to-income ratio",
    r"Change in house prices",
    r"Mortgage rates",
    r"Private housing starts",
    r"Household financial obligations ratio",
    r"Delinquency rates on selected types of household loans",
    r"Change in real business fixed investment",
    r"Change in real business inventories",
    r"Before-tax profits of nonfinancial corporations.*",
    r"Selected components of net business financing.*",
    r"Financing gap and net equity retirement.*",
    r"Net percentage of domestic banks tightening.*",
    r"Default rate on outstanding corporate bonds",
    r"Federal receipts and expenditures",
    r"Net saving",
    r"Change in real government expenditures.*",
    r"Federal government debt held by the public",
    r"Treasury securities held by foreign investors.*",
    r"State and local government net saving",
    r"U\.S\. trade and current account balances",
    r"Prices of oil and of nonfuel commodities",
    r"Prices of major nonfuel commodities",
    r"U\.S\. net financial inflows",
    r"U\.S\. net international securities transactions",
    r"Civilian unemployment rate",
    r"Net change in payroll employment",
    r"Labor force participation rate",
    r"Measures of change in hourly compensation",
    r"Change in PCE prices excluding food and energy",
    r"Alternative measures of price change",
    r"Change in consumer prices.*",
    r"TIPS-based inflation compensation",
    r"Change in unit labor costs",
    r"Interest rates on selected Treasury securities",
    r"Spreads of corporate bond yields over.*",
    r"Implied S&P 500 volatility",
    r"Stock price indexes",
    r"Growth of domestic nonfinancial debt",
    r"M2 growth rate",
    r"Equity indexes in selected foreign industrial countries",
    r"Spread on internationally issued sovereign debt.*",
    r"Official interest rates in selected foreign industrial countries",
    r"U\.S\. dollar nominal exchange rate.*",
    r"U\.S\. dollar exchange rate against.*",
    r"Equity indexes in selected emerging-market economies",
]
_CS_LINE = re.compile("(?:%s)" % "|".join("(?:%s)" % p for p in _CS_LINE_PATTERNS))

# Remove remaining NOTE/SOURCE lines, these are often citations, footnotes, or references, not substantive content. When one of
# these openers is seen (and drop_notes=True), the line AND its wrapped continuation lines are skipped until the next blank line,
# my previous version only removed the first line of a note and left the rest of it behind in the corpus.
_RE_NOTE_START = re.compile(r"(?:NOTE|NOTES|SOURCE|SOURCES|N\.B\.)\s*[:.\u2014-]", re.IGNORECASE)
# "12. See the minutes of the March meeting..." is the classic numbered footnote body. Caveat: this would also remove a numbered
# list whose items start with a capitalized sentence, those are essentially absent from the FED reports, but drop_notes can be
# flipped off if that ever matters for another corpus.
_RE_FOOTNOTE_START = re.compile(r"\d{1,2}\.\s+[A-Z\u201c\"(]")

# Line-level filtering, this entails dropping chart data lines, these are generalizations of my previous per-pattern checks.
_RE_NUMERIC_JUNK = re.compile(r"[\d\s()+\-*/=.:,%$\u2013\u2014]+")   # Lines with only numbers, spaces, and punctuation.
_RE_QUARTER_TICKS = re.compile(r"(?:[QH][1-4]\s*)+", re.IGNORECASE)  # Axis tick rows like "Q1 Q2 Q3 Q4".
_RE_MONTH_TICKS = re.compile(                                        # Axis tick rows like "Jan. Apr. July Oct.".
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]{0,6}\.?\s*)+",
    re.IGNORECASE,
)
_RE_SHORT_ACRONYM = re.compile(r"[A-Z]{1,6}\d{0,2}")                 # Bare series labels standing alone, "GDP", "PCE", "M2".
_RE_ALLCAPS_LINE = re.compile(r"[A-Z][A-Z ,.:;&'()\d-]{7,}")         # Remove any boilerplate that got split into ALL-CAPS running heads.

# Recognizes a line that ends like a finished sentence, allowing for closing quotes/brackets after the terminal punctuation.
_RE_SENTENCE_END = re.compile("[.!?][)\\]\"'\u201d\u2019]*$")


def _is_junk_line(line):
    # Skips lines that are pure chart noise, the structural (non-lexical) checks for one stripped line.
    # Skips lines with only numbers, spaces, and punctuation (this also covers the year rows, "+/-" rows, and math-operator rows
    # which I previously matched with separate patterns).
    if _RE_NUMERIC_JUNK.fullmatch(line):
        return True
    # Skips the quarter and month axis tick rows.
    if _RE_QUARTER_TICKS.fullmatch(line) or _RE_MONTH_TICKS.fullmatch(line):
        return True
    # Skips bare acronym lines ("GDP", "M2") which are chart series labels, while "M2 growth remained moderate." in a sentence
    # survives because it is not the whole line.
    if len(line) <= 8 and _RE_SHORT_ACRONYM.fullmatch(line):
        return True
    # Alpha ratio filter (aggressive for chart-heavy MPRs).
    # Filters lines based on letter-to-character ratio, lines with very few letters, i.e. less than 15% letters, are likely chart
    # data or numerical artifacts.
    alpha = sum(1 for c in line if c.isalpha())
    if alpha < 3:
        return True
    # The isascii() check ensures this only applies to ASCII lines (English text), so non-Latin text is never penalized.
    if line.isascii() and alpha / len(line) < 0.15:
        return True
    return False


def _filter_lines(text, drop_notes, drop_allcaps, min_repeat):
    lines = [line.strip() for line in text.split("\n")]

    # Frequency-based running header/footer removal. "Monetary Policy Report ▪ July 2019" style headers repeat on every single
    # page, so any short line that repeats min_repeat or more times and does not end like a sentence is dropped. This is what makes
    # the cleaner year- and report-agnostic, instead of depending on my hardcoded lists alone, a header format I have never seen
    # before will still be caught simply because it repeats.
    counts = Counter(line for line in lines if line)

    kept = []
    in_note_block = False
    prev_kept = ""
    for line in lines:
        if not line:
            in_note_block = False
            prev_kept = ""
            # Blank lines are paragraph boundaries and are kept, my previous version dropped every empty line, which destroyed the
            # paragraph structure that the reflow further down now depends on.
            kept.append("")
            continue
        # While inside a NOTE/SOURCE/footnote block, every wrapped continuation line is skipped until the next blank line.
        if in_note_block:
            continue
        # Footnote bodies ("12. See the minutes...") are only treated as such at a paragraph start or after a finished sentence,
        # otherwise a body line that happens to wrap right before a small number ("...met on May\n14. The Committee...") would be
        # eaten by mistake.
        looks_like_footnote = (
            _RE_FOOTNOTE_START.match(line)
            and (not prev_kept or _RE_SENTENCE_END.search(prev_kept))
        )
        if drop_notes and (_RE_NOTE_START.match(line) or looks_like_footnote):
            in_note_block = True
            continue
        # The frequency filter described above.
        if (len(line) <= 90 and counts[line] >= min_repeat
                and not _RE_SENTENCE_END.search(line)):
            continue
        # The full-line boiler plate and chart-title lists.
        if _CI_LINE.fullmatch(line) or _CS_LINE.fullmatch(line):
            continue
        # Remove any boilerplate that got split across lines as ALL-CAPS running heads, the two-word minimum stops it from eating
        # bare acronyms (those are handled with their own rule in _is_junk_line).
        if drop_allcaps and " " in line and _RE_ALLCAPS_LINE.fullmatch(line):
            continue
        # The structural chart-noise checks.
        if _is_junk_line(line):
            continue
        kept.append(line)
        prev_kept = line
    return "\n".join(kept)


# Only newline-hyphenation is joined here. My previous rule (\w+)-\s+(\w+) also merged hyphen + ordinary space, which corrupted the
# suspended hyphens that are everywhere in FED prose, "short- and long-term rates" became "shortand long-term rates". Same-line
# hyphens are now left completely alone.
_RE_LINEBREAK_HYPHEN = re.compile(r"(\w+)-\n(\w+)")
# The words that follow a suspended hyphen, "short- and", "one- to three-year", these must NOT be glued onto the previous word.
_SUSPENDED_HYPHEN_WORDS = {"and", "or", "to", "the"}


def _dehyphenate(match):
    left, right = match.group(1), match.group(2)
    # Preserves the suspended hyphen construction, "short-\nand" -> "short- and".
    if right.lower() in _SUSPENDED_HYPHEN_WORDS:
        return left + "- " + right
    # Keeps the hyphen when the second half is capitalized, "non-\nOPEC" -> "non-OPEC".
    if right[0].isupper():
        return left + "-" + right
    # Keeps the hyphen for numeric ranges split at a line break, "2-\n3 percent" -> "2-3 percent".
    if right[0].isdigit():
        return left + "-" + right
    # The standard page-break case, "communi-\ncation" -> "communication".
    return left + right


def _reflow(text, wrap_width=45):
    # Unwraps the hard line breaks inside paragraphs so the LLM sees natural prose. The heuristics for merging a line with the one
    # after it are: the next line starts with a lowercase letter (the strongest signal), or the current line ends with a
    # comma/semicolon, or the current line is at least wrap_width characters and does not end like a sentence (the typical
    # mid-paragraph wrap), including when the next line starts with a digit ("...rose\n2.5 percent"). Headings ("Economic and
    # Financial Developments") are short and are followed by a capitalized line, so they keep their own line. wrap_width defaults
    # low because the MPR PDFs are two-column and extract at roughly 45-60 characters per line.
    merged = []
    current = ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            # A blank line closes the current paragraph.
            if current:
                merged.append(current)
                current = ""
            merged.append("")
            continue
        if not current:
            current = line
            continue
        join = (
            line[0].islower()
            or line[0] in ",;)"
            or current.endswith((",", ";"))
            or (len(current) >= wrap_width and not _RE_SENTENCE_END.search(current))
        )
        if join:
            current = current + " " + line
        else:
            merged.append(current)
            current = line
    if current:
        merged.append(current)
    return "\n".join(merged)


# In-text footnote markers that PDF extraction leaves glued onto sentences, "...policy rates.12 The Committee" ->
# "...policy rates. The Committee". The fixed-width lookbehind requires a letter-or-")" followed by punctuation, therefore decimals
# ("1.5 percent") and series codes ("M2.") are never touched.
_RE_FOOTNOTE_MARKER = re.compile(r"(?<=[a-z)][.;:,])\d{1,2}(?=\s|\Z)")

# Clean up whitespace and fix text artifacts left behind by the extraction, stray spaces before punctuation and inside brackets.
_RE_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?%])")
_RE_SPACE_AFTER_OPEN = re.compile(r"([(\[])\s+")
_RE_SPACE_BEFORE_CLOSE = re.compile(r"\s+([)\]])")


def clean_text(text, drop_notes=True, reflow=True, drop_allcaps=True,
               min_repeat=3, min_alpha_chars=0):
    # Cleans one scraped FED document ready for LLM continued pre-training.
    #   drop_notes      - removes NOTE./SOURCE. blocks, numbered footnote bodies (including their wrapped continuation lines), and
    #                     the in-text footnote markers ("rates.12" -> "rates.").
    #   reflow          - unwraps the hard line breaks so the model sees whole paragraphs, this is arguably the single biggest win
    #                     for pre-training quality.
    #   drop_allcaps    - drops ALL-CAPS lines of two or more words (the running heads in the older reports), set this to False to
    #                     keep ALL-CAPS section headings instead.
    #   min_repeat      - a short line repeated this many times within one document is treated as a running header/footer.
    #   min_alpha_chars - if the cleaned text has fewer alphabetic characters than this, "" is returned so I can drop the document
    #                     from the corpus at the call site.
    if not text:
        return ""

    # Unicode normalization is the initial step, as described at the top of the module.
    text = _normalize_unicode(text)

    # Line-level boiler plate and chart debris, run before the de-hyphenation so a running header sitting between the two halves
    # of a hyphenated word is gone before the join is attempted.
    text = _filter_lines(text, drop_notes, drop_allcaps, min_repeat)

    # Joins words that were split by a hyphen at line breaks.
    text = _RE_LINEBREAK_HYPHEN.sub(_dehyphenate, text)

    # Paragraph reflow, unwrapping the hard line breaks left by the PDF extraction.
    if reflow:
        text = _reflow(text)

    if drop_notes:
        # Strips the in-text footnote markers.
        text = _RE_FOOTNOTE_MARKER.sub("", text)
    # Tightens the stray spaces before punctuation and inside brackets.
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _RE_SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _RE_SPACE_BEFORE_CLOSE.sub(r"\1", text)
    # Clean up whitespace and fix text artifacts, replaces multiple whitespace characters (spaces, tabs, formfeeds, carriage
    # returns) with a single space.
    text = re.sub(r"[ \t\f\r]+", " ", text)
    # Removes spaces around newlines.
    text = re.sub(r" *\n *", "\n", text)
    # Replaces 3 or more consecutive newlines with 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Drops documents which come out essentially empty after cleaning, the caller can filter on the empty string.
    if min_alpha_chars and sum(1 for c in text if c.isalpha()) < min_alpha_chars:
        return ""
    return text


if __name__ == "__main__":
    # A small worked example which exercises the pipeline end to end, the cover page, a chart block, a running header sitting inside
    # a hyphenated word, a glued fraction, a suspended hyphen, and the NOTE/SOURCE/footnote blocks.
    sample = (
        "For use at 10:00 a.m., EDT\n"
        "February 11, 2004\n\n"
        "Board of Governors of the Federal Reserve System\n"
        "Monetary Policy Report to the Congress\n\n"
        "Selected interest rates\nPercent\n2000 2001 2002 2003\n+ 1.2 - 0.5\n"
        "Ten-year Treasury\n\n"
        "Monetary Policy Report \u25aa February 2004\n"
        "The contents of the report indicate that the Committee raised the\n"
        "target for the federal funds rate to 51\u20444 percent, and the communi-\n"
        "Monetary Policy Report \u25aa February 2004\n"
        "cation of policy intentions improved.1 As shown in\n"
        "figure 2, short-\n"
        "and long-term rates fell. On February 14, 2023, the Committee met on\n"
        "a Wednesday.\n\n"
        "NOTE. Data are quarterly and extend through\n"
        "2003:Q4.\n"
        "SOURCE. Department of Commerce.\n"
        "1. See the minutes of the March meeting for\n"
        "further details.\n"
    )
    print(clean_text(sample))
