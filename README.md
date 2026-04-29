# Legal Citation Sequences & Markov Analysis

Two-stage pipeline for **Philippine legal opinion text**: extract ordered citations with the Gemini API, then build **first-order Markov chains** on normalized citation titles and citation *types* (transitions, synthetic sequences, heatmaps).

---

## What it does

1. **Extraction (`extract.py`)**  
   Reads plain-text decisions from `cases/`, asks Gemini to list citations **verbatim and in document order**, then classifies each item and maps it to a short normalized token (for example, `R51S3` for a Rules of Court reference, `C3S2` for Constitution Article/Section patterns).

2. **Markov modeling (`markov.py`)**  
   Loads per-case sequences from JSON, estimates transition probabilities **within each case only** (no artificial edges between the last citation of one opinion and the first of the next), and supports:
   - Top-*k* next states from a given citation or type  
   - Random walk sequence generation  
   - Multi-step transition matrices (e.g. M², M³) and heatmaps  

Precomputed chains are stored as `title_markov_chain.json` and `type_markov_chain.json` so you can reload without rebuilding.

---

## Repository layout

| Path | Role |
|------|------|
| `cases/*.txt` | Source opinions (one file per case). |
| `extract.py` | LLM extraction + classification → sequence JSON files. |
| `markov.py` | Build/load chains, matrices, features, heatmaps. |
| `title_sequence.json`, `types_sequence.json` | Nested lists: one sequence per case. |
| `title_sequence_flat.json`, `types_sequence_flat.json` | Flattened sequences (compatibility / aggregation). |
| `title_markov_chain.json`, `type_markov_chain.json` | Saved transition probabilities. |
| `markov_chain.json` | Legacy or alternate chain export (see script usage). |
| `titles_heatmap.jpg`, `types_heatmap.jpg` | Example 1-step matrix heatmaps. |
| `markov_chain_explanation.txt` | High-level narrative of extraction vs. computation. |

The sample cases are **Supreme Court of the Philippines**–style materials (e.g. `G.R. No. …`, Rule 45, civil law citations).

---

## Requirements

- **Python 3.10+** recommended (uses `ast.literal_eval`, type hints).  
- **Google Gemini API key** for extraction.

### Python packages

Install the libraries used by the scripts:

```bash
pip install google-genai numpy pandas matplotlib seaborn
```

`extract.py` uses the **`google-genai`** SDK (`from google import genai`).  
`markov.py` uses **NumPy**, **Pandas**, **Matplotlib**, and **Seaborn**.

A small `package.json` is present with `@google/genai` for JavaScript tooling; the Python pipeline does not require Node.js unless you add your own JS tooling.

---

## Configuration

### API key

Set your Gemini key before running extraction:

**Windows (PowerShell)**

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

**Linux / macOS**

```bash
export GEMINI_API_KEY="your-key-here"
```

If `GEMINI_API_KEY` is missing, `extract.py` raises a clear error.

### Model

`extract.py` calls `gemini-2.5-flash`. To use another model, edit the `model=` argument in `generate_citations()`.

---

## Workflow

### 1. Prepare case files

Place UTF-8 `.txt` files under `cases/`. Files are processed in sorted path order (e.g. `case1.txt`, `case2.txt`, …).

### 2. Run extraction

From the project root:

```bash
python extract.py
```

This will:

- Call Gemini for each case with a strict prompt: only citations from Constitution, statutes, jurisprudence, and administrative rules; preserve order; output a Python-parseable list of strings.
- **Classify** each title into: `Constitution`, `Statute`, `Jurisprudence`, `Administrative rule`, or `Other` (excluded from sequences).
- **Normalize** strings for stable Markov states (e.g. rule/section patterns, RA/PD/BP, Civil Code articles, case citations).

Outputs:

- `title_sequence.json` / `types_sequence.json` — list of per-case lists.  
- `title_sequence_flat.json` / `types_sequence_flat.json` — single concatenated lists.

### 3. Build or load Markov chains

Edit `markov.py` if needed:

```python
mIsNew = 1  # build chains from JSON and overwrite title_markov_chain.json / type_markov_chain.json
mIsNew = 0  # load existing JSON chains
```

Then run:

```bash
python markov.py
```

With `mIsNew = 1`, chains are rebuilt from `title_sequence.json` and `types_sequence.json`. With `mIsNew = 0`, it loads `title_markov_chain.json` and `type_markov_chain.json`.

The default `main()`:

1. Prints full title transition matrix (chunked) and saves `titles_heatmap.jpg`.  
2. Prints type matrix with fixed state order and saves `types_heatmap.jpg`.  
3. Demonstrates **Feature 1** (top-*k* next states) on the title chain for state `R51S3` if data exists.  
4. Demonstrates **Feature 2** (random sequences) on the type chain from the first observed type.  
5. Runs **Feature 3** on the type chain: M¹, M², and M³ with heatmaps for the multi-step matrices.

Heatmaps and `plt.show()` require a display backend suitable for your environment (interactive desktop vs. headless server).

---

## Citation normalization (summary)

Logic lives in `classify_and_normalize()` in `extract.py`:

- **Jurisprudence** — patterns like `G.R. No.`, ` v. ` / `vs.`  
- **Constitution** — `const.`, `constitution`, `article`, with optional `section` → tokens like `C{article}S{section}`.  
- **Statute** — Civil Code articles → `CC{n}`; Acts → `A{n}`; `R.A.` / `PD` / `BP` prefixes normalized.  
- **Administrative rule** — `Rules` + rule number, sections → `R{n}` / `R{n}S{m}`; bare `Section` can inherit the last seen rule number in the same case.

Anything else is **`Other`** and is dropped from the Markov input sequences.

---

## Design notes

- **Sequences are per case** so transitions reflect ordering inside one opinion, not across unrelated documents.  
- The LLM is instructed not to invent citations; downstream code still expects parseable list output.  
- Markov chains are **first-order** (next state depends only on current state).  
- `get_top_k_next_citations` returns the first *k* entries from the transition dict (insertion order); for true probability ranking, sort by probability before slicing if your chain JSON ordering changes.

---

## Further reading

See `markov_chain_explanation.txt` for a prose walkthrough of the extraction phase vs. the Markov computation phase (written when the extractor was referred to as `proj.py`; the implementation file is `extract.py`).

---

## License

No license file is included in this repository. Add one if you intend to distribute or reuse the code.
