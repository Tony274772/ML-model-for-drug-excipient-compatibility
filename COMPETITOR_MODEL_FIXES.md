# Fix List: DE-Compatibility Stacking Model (phuongnvp/drug_exp_compatibility)

## Purpose
This is a fix/prep list for an AI coding agent. Goal: get this repo's exact
published architecture training end-to-end, cleanly, **on our own train/val/test
split**, so its metrics are directly comparable to our MoLFormer-XL model. Do
**not** change the architecture, algorithms, hyperparameter grids, class
weights, or thresholding method — the point is a faithful reproduction, not an
improvement. Only fix bugs, portability issues, and split-fairness issues
listed below.

Source paper: Nguyen et al., *Int. J. Pharmaceutics* 653 (2024) 123884.
Source repo files provided: `2D_descriptors.py`, `mol2vec_descriptors.py`,
`model_selection.py`, `Stack_model.py`, `Combine_mol2vec_and_2D_model.py`,
`model_deploy.py`, `dataset.csv`, `variables.txt`, `requirements.txt`,
pretrained `.pkl` artifacts.

---

## 0. Critical: train/val/test split must be OURS, not a fresh random split

**Every** script in this repo (`model_selection.py`, `Stack_model.py`,
`Combine_mol2vec_and_2D_model.py`) independently calls
`train_test_split(..., test_size=.4)` then `test_size=.5` on the remainder,
with **no `random_state`**, on `dataset.csv` (3544 rows). None of these three
scripts even agree with each other on which rows land in train/val/test.

Our own dataset (`start_dataset.csv`, 3544 pairs) was originally sourced from
this same GitHub repo, and our current split (2030 train / 842 val / 672 test)
sums to 3544 — the same row count. So:

- Do **not** call `train_test_split` at all in the adapted scripts.
- Instead, load our existing train/val/test CSVs (whatever files hold our
  2030/842/672 split) and join them to `dataset.csv` on
  `(API_CID, Excipient_CID)` (fall back to SMILES pair match if any CIDs
  don't align after our dedup/cleaning).
- If some of our pairs aren't present in `dataset.csv` (e.g., we added new
  pairs during cleaning) or vice versa, log the row counts that matched vs.
  didn't, and confirm with me before silently dropping rows.
- All three "phases" (mol2vec model, 2D-descriptor model, combined stacking
  model, threshold selection) must use **the same fixed split**, loaded once
  and reused everywhere.

This is the single most important fix — without it, any metric comparison
between their model and ours is not apples-to-apples.

---

## 1. Broken / crashing bugs (must fix to run at all)

1. **`mol2vec_descriptors.py` reads the wrong filename.**
   `pd.read_csv('data.csv')` — actual file is `dataset.csv`. Fix the path (or
   accept an input path argument).

2. **`mol2vec_descriptors.py` has a hardcoded Windows path.**
   `os.chdir(r'C:\Users\PC\OneDrive - Hanoi University Of Pharmacy\...')` —
   remove entirely. Use relative paths / a config variable for where
   `model_300dim.pkl` lives.

3. **Dead/broken file handle in `mol2vec_descriptors.py`.**
   `open('model_300dim.pkl', 'rb')` is called and the result is discarded
   (never closed, never used) right before the real load via
   `word2vec.Word2Vec.load('model_300dim.pkl')`. Delete the stray `open()`
   call.

4. **`model_selection.py` has a Python syntax error.**
   `df=pd.read_csv('mol2vec_data.csv)` — missing closing quote on the
   filename string. Fix to `pd.read_csv('mol2vec_data.csv')`.

5. **`Combine_mol2vec_and_2D_model.py` line 142: undefined variable.**
   References `thresholds1` (plural) but the variable defined earlier (line
   77) is `threshold1` (singular). Fix the typo — this is the line that
   applies the F1-optimal threshold to the held-out test set, so it's not
   optional.

---

## 2. Silent correctness bugs (will run, but produce wrong numbers)

6. **`model_selection.py` selects the wrong label column.**
   `y = df.iloc[:,-1].values` assumes the last column of `mol2vec_data.csv`
   is `Outcome1`. It isn't. Trace `mol2vec_descriptors.py`'s column
   construction: after dropping the mol/sentence/vector helper columns, the
   remaining original columns keep their original order from `dataset.csv`
   — `API_CID, Excipient_CID, Outcome1, API_Smiles, Excipient_Smiles` — so
   the **last** column is `Excipient_Smiles`, not `Outcome1`. Fix to select
   `y = df['Outcome1'].values` by name (this is exactly what `Stack_model.py`
   does correctly — mirror that).

7. **`2D_descriptors.py` / RDKit descriptor count is version-dependent —
   never index descriptor columns positionally.**
   `Descriptors._descList` is not a stable, version-independent list; the
   number and order of descriptors changes across RDKit releases. The
   `variables.txt` file shipped in this repo captures the *exact* named
   feature set actually used to train the shipped `.pkl` models: 201 API-side
   columns + 197 excipient-side columns (398 total) — note only 197 not 201
   on the excipient side, because 4 partial-charge descriptors
   (`MaxPartialCharge`, `MinPartialCharge`, `MaxAbsPartialCharge`,
   `MinAbsPartialCharge`) are absent for excipients, almost certainly because
   they error out (`NaN`/exception) on the metal salts and inorganic oxides
   in this dataset (e.g. `O=[Ti]=O`, `O=[Pb]=O`) and got dropped downstream.
   **Always** select columns by name from `variables.txt`, exactly like
   `model_deploy.py` already does (`X_2D = data_2D[selected_descriptors]`) —
   never by positional slice.

8. **`Combine_mol2vec_and_2D_model.py` uses a hardcoded positional slice for
   the 2D features.**
   `X_2D = df1.iloc[:,1:399].values` assumes exactly 398 usable descriptor
   columns starting at column index 1. On any current RDKit version this is
   wrong (current RDKit ships ~217 descriptors/molecule, i.e. ~434 raw
   columns before pruning to the named 398). Replace with named-column
   selection via `variables.txt`, consistent with fix #7.

9. **Filename mismatch for saved 2D/mol2vec models.**
   `Combine_mol2vec_and_2D_model.py` loads `2D_model.pkl` and
   `mol2vec_model.pkl`, but the repo actually ships `model_2D.pkl` and
   `model_mol2vec.pkl`. Align the filenames (rename on save, or fix the load
   calls) — pick one convention and use it everywhere including
   `model_deploy.py` (which correctly uses `model_2D.pkl` / `model_mol2vec.pkl`
   already, so standardize on that).

10. **Missing imputation for failed descriptors.**
    `getMolDescriptors()` sets any descriptor that throws an exception to
    `None`. Nothing downstream imputes these before model fitting. Add an
    explicit imputation step (e.g. `SimpleImputer(strategy='median')` fit on
    train only, applied to val/test) right after descriptor computation, and
    log how many `NaN`s appear per column/molecule so we know how much of the
    2D-descriptor branch is actually real signal for the inorganic
    excipients.

---

## 3. Missing pieces (not bugs, but the repo is incomplete for full reproduction)

11. **No training script for the 2D-descriptor base model exists.**
    `2D_descriptors.py` only builds features; `Combine_mol2vec_and_2D_model.py`
    only *loads* an already-trained `model_2D.pkl`. There's no equivalent of
    `Stack_model.py` for the 2D branch. Write one: same structure as
    `model_selection.py` (grid search over the same 8 algorithms) →
    `Stack_model.py` (stack the best 3 — matching the paper's methodology —
    with `StackingClassifier(estimators=[AdaBoost, RandomForest, XGBoost],
    final_estimator=LogisticRegression)`, and apply the same 0.3/0.7 class
    weighting via `sample_weight`) but on the 2D-descriptor feature matrix
    instead of the mol2vec one. Save the result as `model_2D.pkl`.

12. **Class weighting is only wired up for the mol2vec stack.**
    `Stack_model.py` applies `sample_weight=[{0:0.3,1:0.7}[i] for i in
    y_train_resampled]` when fitting the mol2vec `StackingClassifier`. This
    same weighting must be replicated in the new 2D-branch training script
    (task 11) for a faithful reproduction of the paper's reported 0.3:0.7
    result (Table 3).

13. **`ghostml` is used but never declared as a dependency.**
    `Combine_mol2vec_and_2D_model.py` does `import ghostml` for the GHOST
    threshold-optimization step, but `requirements.txt` doesn't list it. Add
    it (`pip install ghostml`, or if unavailable on PyPI, install from
    `https://github.com/rinikerlab/GHOST`).

14. **Custom mol2vec fork required.**
    `requirements.txt` pins `git+https://github.com/phuongnvp/mol2vec`, not
    the public `mol2vec` PyPI package. Install this exact fork — the public
    package's `mol2alt_sentence`/`sentences2vec` API may not match what these
    scripts expect.

15. **Filename `model_300dim.pkl` is misleading.**
    Despite the name, this is the pretrained mol2vec **word2vec embedding
    model producing 100-dimensional vectors per molecule** (confirmed by the
    paper text and by `X1`/`X2` each being 100-wide, concatenated to 200
    total features per pair). Don't assume 300-dim anywhere in the pipeline;
    verify empirically after loading (`w2vec_model.wv.vector_size == 100`).

---

## 4. Reproducibility fixes

16. **No `random_state` anywhere.**
    `train_test_split`, `SVMSMOTE`, `RandomForestClassifier`,
    `MLPClassifier`, `DecisionTreeClassifier`, `AdaBoostClassifier`,
    `XGBClassifier`, `KNeighborsClassifier`, `LogisticRegression`,
    `StackingClassifier` — none of these are seeded, so results differ every
    run. Add `random_state=42` (or one shared seed variable) to every
    estimator and to `SVMSMOTE`. (Split randomness is moot once fix #0 is
    applied, but keep the seed for the SMOTE oversampling and every model's
    internal randomness.)

17. **No stratification (moot after fix #0, but note for any script that
    still does its own splitting, e.g. if we ever need to reproduce their
    *original* random-split numbers for a sanity check).** Add
    `stratify=y` given the ~9–10% positive rate.

---

## 5. Robustness fixes (long unattended run — this will take a while)

18. **No per-molecule error handling around SMILES parsing in
    `2D_descriptors.py`.**
    `Chem.MolFromSmiles(smiles)` can return `None` for malformed/disconnected
    SMILES, and `getMolDescriptors(None, ...)` will crash the whole loop
    partway through. Wrap parsing in try/except, log the offending row
    (API_CID/Excipient_CID + SMILES), and either skip or impute — don't let
    one bad excipient kill an hours-long featurization run.

19. **No checkpointing during featurization.**
    Both `2D_descriptors.py` and `mol2vec_descriptors.py` process the full
    dataset in one in-memory pass with no incremental save. Add periodic
    checkpoint writes (e.g. every 500 rows) to a resumable CSV/parquet file,
    consistent with the checkpoint pattern already used in our own
    `verify_fetch_cid.py`/`fetch_api_cid.py` scripts, so a crash on row 3000
    doesn't require recomputing rows 1–2999.

20. **`AddHs()` applied before mol2vec featurization — keep this, but log
    the UNK-fragment rate.**
    Both `mol2vec_descriptors.py` and `model_deploy.py` call `Chem.AddHs()`
    before `mol2alt_sentence`. This is consistent between training and
    deployment (so it's not a bug to "fix"), but explicit hydrogens change
    the Morgan-environment identifiers fed into mol2vec, which can increase
    how often `sentences2vec(..., unseen='UNK')` falls back to the generic
    UNK vector for fragments not in the pretrained vocabulary. Add a log
    line reporting the fraction of substructure tokens per molecule that hit
    `UNK`, for both API and excipient sentences — useful diagnostic context
    when comparing against our MoLFormer-XL encoder, which doesn't have this
    failure mode.

21. **Don't try to reuse the shipped `.pkl` files.**
    `model_2D.pkl`, `model_mol2vec.pkl`, and `model_lr.pkl` were pickled
    under scikit-learn 1.2.2. On any current scikit-learn they fail to
    unpickle (`ValueError: node array from the pickle has an incompatible
    dtype` — sklearn's Cython tree-node struct changed when
    `missing_go_to_left` support was added in sklearn ≥1.3). This confirms
    a full retrain from scratch is necessary anyway — don't waste time
    attempting a "warm start" from these artifacts.

---

## 6. Progress / logging requirements (explicit ask)

Add clear, unambiguous progress indicators throughout — this will run
unattended for a while (8-algorithm grid search × 2 feature branches, plus
featurization of 3544 pairs):

- Wrap the `2D_descriptors.py` and `mol2vec_descriptors.py` per-molecule
  loops in `tqdm` progress bars (already a pinned dependency —
  `tqdm==4.65.0`).
- Set `GridSearchCV(..., verbose=2)` (not `verbose=0` as in the original) for
  every one of the 8 algorithms in `model_selection.py`, for both the
  mol2vec and 2D branches.
- Print/log a clear header before each stage starts and a summary line when
  it finishes, e.g.:
  `[1/8] Grid search: Random Forest (mol2vec features)... done in 42.3s — best acc 0.96`
- Log dataset shapes at every major step (raw load, post-split, post-SMOTE,
  post-imputation) so a silent shape mismatch is caught immediately.
- Log final metrics (accuracy, precision, recall, F1, AUC, MCC) to a single
  results CSV/JSON per branch (mol2vec-only, 2D-only, combined-stack,
  threshold-tuned) so they can be pulled directly into our comparison table
  against our model — don't just `print()` and lose them.
- At the very end, print a single consolidated summary table (branch ×
  metric) mirroring the paper's Table 2/3/4/5 layout, so the final
  competitor numbers on **our split** are easy to line up against our
  MoLFormer-XL results.

---

## 7. What NOT to change

To keep this a fair architecture-vs-architecture comparison, do not:
- Change the 8-algorithm grid search grids/ranges in `model_selection.py`.
- Change the stacking structure (AdaBoost + RandomForest + XGBoost →
  Logistic Regression meta-learner).
- Change SVM-SMOTE for a different resampler.
- Change the 0.3:0.7 class weighting.
- Change the F1-based threshold search (range `0, 1, step 0.001`) or drop the
  GHOST-algorithm comparison.
- Add new features (e.g. don't add RDKit descriptors to the mol2vec branch,
  don't add 3D descriptors — they're mentioned in the paper but never
  actually implemented in this repo, so the "combined model" being
  reproduced here is specifically **mol2vec + 2D-descriptors**, matching
  Table 5's F1=0.87 result, not the paper's abstract headline numbers, which
  also folded in ECFP/3D variants that aren't in this codebase).
