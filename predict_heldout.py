"""Run inference on new_heldout/heldout.csv and save predictions.

This mirrors the pipeline used in predict_validation_pairs.py:
- 2D RDKit descriptors selected by variables.txt
- mol2vec features for API and Excipient SMILES
- stacked model: 2D branch + mol2vec branch -> LogisticRegression meta-classifier
- Uses the retrained competitor model artifacts in competitor_results/
- Saves predictions with logit, prob, and pred (threshold 0.5)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import word2vec
from mol2vec.features import MolSentence, mol2alt_sentence, sentences2vec
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = ROOT / "new_heldout" / "heldout.csv"
DEFAULT_OUTPUT_PATH = ROOT / "new_heldout" / "heldout_with_preds.csv"
COMPETITOR_DIR = ROOT / "competitor_results"


def parse_mol_robust(smiles: str) -> Chem.Mol | None:
    """Parse SMILES with standard method, falling back to sanitize=False if needed."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
            return mol
    except Exception:
        pass
    return None


def get_mol_descriptors(mol: Chem.Mol, missing_val=np.nan) -> dict:
    """Compute all RDKit descriptors for a molecule."""
    result = {}
    for name, fn in Descriptors._descList:
        try:
            value = fn(mol)
        except Exception:
            value = missing_val
        result[name] = value
    return result


def load_first_existing(paths: list[Path]):
    for path in paths:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(f"None of these files exist: {paths}")


def load_models():
    """Load the retrained model artifacts for this repo."""
    model_2d = load_first_existing(
        [
            COMPETITOR_DIR / "model_2D_retrained.pkl",
            ROOT / "model_2D.pkl",
        ]
    )
    model_mol2vec = load_first_existing(
        [
            COMPETITOR_DIR / "model_mol2vec_retrained.pkl",
            ROOT / "model_mol2vec.pkl",
        ]
    )
    model_lr = load_first_existing(
        [
            COMPETITOR_DIR / "model_lr_retrained.pkl",
            ROOT / "model_lr.pkl",
        ]
    )
    imputer_2d = None
    for candidate in [COMPETITOR_DIR / "imputer_2D.pkl"]:
        if candidate.exists():
            imputer_2d = joblib.load(candidate)
            break

    return model_2d, model_mol2vec, model_lr, imputer_2d


def build_features(df: pd.DataFrame, selected_descriptors: list[str], mol2vec_model):
    """Build aligned 2D and mol2vec features for rows with valid SMILES."""
    # Detect drug and excipient column names
    col_map = {c.lower(): c for c in df.columns}
    drug_col = col_map.get("drugs") or col_map.get("api_smiles") or col_map.get("drug")
    exc_col = col_map.get("excipients") or col_map.get("excipient_smiles") or col_map.get("excipient")

    if not drug_col or not exc_col:
        raise ValueError(
            f"Could not identify drug and excipient SMILES columns in DataFrame columns: {df.columns.tolist()}"
        )

    logging.info("Using '%s' as API SMILES and '%s' as Excipient SMILES", drug_col, exc_col)

    valid_indices = []
    x_2d_rows = []
    x_mol2vec_rows = []

    for idx, row in df.iterrows():
        api_smiles = str(row[drug_col])
        excipient_smiles = str(row[exc_col])

        mol_api = parse_mol_robust(api_smiles)
        mol_excipient = parse_mol_robust(excipient_smiles)

        if mol_api is None or mol_excipient is None:
            logging.warning("Skipping row %s due to unparseable SMILES.", idx)
            continue

        try:
            api_desc = get_mol_descriptors(mol_api)
            excipient_desc = get_mol_descriptors(mol_excipient)

            row_2d = {}
            row_2d.update(api_desc)
            row_2d.update({f"{name}_exp": value for name, value in excipient_desc.items()})
            x_2d_rows.append([row_2d.get(name, np.nan) for name in selected_descriptors])

            mol_api_h = Chem.AddHs(mol_api)
            mol_exc_h = Chem.AddHs(mol_excipient)
            api_sentence = MolSentence(mol2alt_sentence(mol_api_h, 1))
            excipient_sentence = MolSentence(mol2alt_sentence(mol_exc_h, 1))
            api_vec = np.asarray(sentences2vec([api_sentence], mol2vec_model, unseen="UNK")[0], dtype=float)
            excipient_vec = np.asarray(sentences2vec([excipient_sentence], mol2vec_model, unseen="UNK")[0], dtype=float)
            x_mol2vec_rows.append(np.concatenate([api_vec, excipient_vec], axis=0))
            valid_indices.append(idx)
        except Exception as exc:
            logging.warning("Skipping row %s due to feature extraction error: %s", idx, exc)

    x_2d = np.asarray(x_2d_rows, dtype=float)
    x_mol2vec = np.asarray(x_mol2vec_rows, dtype=float)
    return valid_indices, x_2d, x_mol2vec


def main():
    parser = argparse.ArgumentParser(description="Run inference on heldout.csv")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to heldout.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to output csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows from %s", len(df), args.input)

    config_path = ROOT / "variables.txt"
    with config_path.open("r", encoding="utf-8") as handle:
        selected_descriptors = [name for name in handle.read().strip().split(",") if name]
    logging.info("Loaded %d selected 2D descriptors from variables.txt", len(selected_descriptors))

    mol2vec_model_path = ROOT / "model_300dim.pkl"
    if not mol2vec_model_path.exists():
        raise FileNotFoundError(f"Mol2vec embedding model not found: {mol2vec_model_path}")
    mol2vec_model = word2vec.Word2Vec.load(str(mol2vec_model_path))
    logging.info("Loaded mol2vec embedding model: vector_size=%d", mol2vec_model.wv.vector_size)

    model_2d, model_mol2vec, model_lr, imputer_2d = load_models()
    logging.info("Loaded models from %s", COMPETITOR_DIR if COMPETITOR_DIR.exists() else ROOT)

    valid_indices, x_2d, x_mol2vec = build_features(df, selected_descriptors, mol2vec_model)
    logging.info("Feature matrix shapes - 2D: %s, mol2vec: %s", x_2d.shape, x_mol2vec.shape)

    if len(valid_indices) == 0:
        raise RuntimeError("No valid rows were found in input file.")

    if imputer_2d is not None:
        x_2d = imputer_2d.transform(x_2d)
        logging.info("Applied 2D imputer from competitor_results/imputer_2D.pkl")
    elif np.isnan(x_2d).any():
        logging.warning("No 2D imputer found; filling remaining NaNs with 0.0.")
        x_2d = np.nan_to_num(x_2d, nan=0.0)

    prob_2d = model_2d.predict_proba(x_2d)[:, 1]
    prob_mol2vec = model_mol2vec.predict_proba(x_mol2vec)[:, 1]
    meta_features = np.column_stack([prob_2d, prob_mol2vec])

    logit = model_lr.decision_function(meta_features)
    prob = 1.0 / (1.0 + np.exp(-logit))
    pred = (prob >= 0.5).astype(int)

    out_df = df.copy()
    out_df["logit"] = np.nan
    out_df["prob"] = np.nan
    out_df["pred"] = np.nan

    out_df.loc[valid_indices, "logit"] = logit
    out_df.loc[valid_indices, "prob"] = prob
    out_df.loc[valid_indices, "pred"] = pred

    # Save to primary output path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    logging.info("Wrote predictions to %s", args.output)

    # Also save a copy to outputs folder if different
    outputs_copy = ROOT / "outputs" / "heldout_with_preds.csv"
    if args.output != outputs_copy:
        outputs_copy.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(outputs_copy, index=False)
        logging.info("Also wrote duplicate copy to %s", outputs_copy)

    logging.info("Valid rows predicted: %d/%d", len(valid_indices), len(df))

    # Evaluate metrics if ground truth column is present
    target_col = None
    for candidate in ["Output Value", "output_value", "ground_truth", "Outcome1", "outcome"]:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is not None:
        y_true = out_df.loc[valid_indices, target_col].values
        y_pred = out_df.loc[valid_indices, "pred"].values.astype(int)
        y_prob = out_df.loc[valid_indices, "prob"].values

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_prob)
        loss = log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15))
        cm = confusion_matrix(y_true, y_pred)

        logging.info("\n" + "=" * 50)
        logging.info("EVALUATION ON HELDOUT SET (threshold=0.5):")
        logging.info("=" * 50)
        logging.info("Accuracy:   %.4f", acc)
        logging.info("Precision:  %.4f", prec)
        logging.info("Recall:     %.4f", rec)
        logging.info("F1 Score:   %.4f", f1)
        logging.info("MCC:        %.4f", mcc)
        logging.info("PR-AUC:     %.4f", pr_auc)
        logging.info("Log Loss:   %.4f", loss)
        logging.info("Confusion Matrix [TN, FP / FN, TP]:\n%s", cm)
        logging.info("=" * 50)


if __name__ == "__main__":
    main()
