# README — `Central_Banker_Speeches_extraction_from_BIS_Frame.ipynb`

**Role in the project:** corpus selection. From the BIS data portal's continuously updated dump of all central-bank speeches, keep only Federal Reserve, ECB, and Banque de France speeches and save each to its own CSV [stated in the notebook's opening markdown]. This file is the upstream source of `fed_speeches.csv` consumed by `Fed_Speech_Cleaner.ipynb`.

**Evidence tags used below:** **[stated]** = written in the code/comments · **[inferred]** = most plausible reading, verify · **[recorded]** = taken verbatim from the saved notebook outputs (last executed run).

---

## Input

`speeches.csv` (BIS dump) — 20,728 rows × 6 columns: `url, title, description, date, text, author` [recorded]. The dump is a **living dataset**: it is updated continuously, and the saved run even shows description lines dated into 2026–2027 [recorded]. All counts below are snapshot-specific — archive the exact file (with a SHA-256 hash) used for the published results.

## Outputs

Written to `…/Data Folders/filtered_speeches/` with `index=False`, after dropping the helper column:

| File | Rows [recorded] |
|---|---:|
| `fed_speeches.csv` | 2,671 |
| `ecb_speeches.csv` | 2,821 |
| `banque_de_france_speeches.csv` | 556 |

## Method, decision by decision

1. **Build an `institution_segment` column.** For each `description`, the regex

   ```
   ^(.*?),\s*(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.|H\.E\.|M\.|Mme\.)
   ```

   extracts everything before the first honorific — "the 'role + institution' portion of the description, i.e. everything before the speaker's name/title marker" [stated]. If no honorific is present (e.g., press releases with no named speaker), the **whole description** is used as a fallback [stated].

   *Why restrict matching to this segment* [inferred, high confidence]: institution keywords must identify the **speaker's** institution. Descriptions routinely name other institutions as hosts or venues ("…before the Association of German Mortgage Banks in Frankfurt…"), so matching the full description would mis-assign such speeches. The honorific list includes the French `M.` / `Mme.` because Banque de France speeches are in scope [inferred].

   *Known limitation:* a description whose speaker name carries no honorific falls back to full-description matching and could re-admit a venue-based false positive. Mitigated by decision 2.

2. **Keyword filters — "specific phrases only, no bare substrings"** [stated]. Case-insensitive, word-boundary-anchored alternations applied to `institution_segment`:

   - Fed: `\bfederal reserve\b`, `\bboard of governors\b`
   - ECB: `\beuropean central bank\b`, `\bECB\b`
   - BdF: `\bbanque de france\b`, `\bbank of france\b`

   The `\b` anchors prevent partial-word hits; the phrase-level design prevents single generic words ("bank", "reserve") from firing [inferred from the "no bare substrings" comment].

3. **Eyeball check.** The first five sample descriptions per subset are printed and were inspected [recorded] — the printed samples are consistent with the intended institutions.

## Reproducibility

- **Deterministic** — no RNG anywhere in this notebook. Identical input file ⇒ identical outputs.
- **Environment:** Python 3.13.12, Jupyter kernel `base` [recorded in metadata]; uses `pandas`, `numpy` (imported; `numpy` unused), `re`, `pathlib`. No versions pinned — pin before submission.
- **Hard-coded absolute path hazard:** the input/output root is `/Users/lorenzouberti/Desktop/BANQUE DE FRANCE /Data Folders/` — note the **trailing space** in `BANQUE DE FRANCE ` which must be reproduced verbatim if the tree is mirrored. Prefer switching to repo-relative paths.

## Audit notes / pre-submission checks

- The three subsets are **not guaranteed disjoint**: a description whose institution segment names two institutions (joint events, ECB officials described alongside a national central bank) lands in both files. Check pairwise intersections on `url` before describing the subsets as partitions.
- Input-drift: re-downloading `speeches.csv` will change the counts. Snapshot + hash the input.
