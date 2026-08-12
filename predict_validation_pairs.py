"""Run inference on outputs/validation_pairs.csv and save predictions.

This script mirrors the local competitor pipeline:
- 2D RDKit descriptors selected by variables.txt
- mol2vec features for API and excipient SMILES
- stacked final LogisticRegression model

It prefers the retrained artifacts under competitor_results/ when present and
falls back to the root-level model files otherwise.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import word2vec
from mol2vec.features import MolSentence, mol2alt_sentence, sentences2vec
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "outputs" / "validation_pairs.csv"
OUTPUT_PATH = ROOT / "outputs" / "validation_pairs_with_preds.csv"
COMPETITOR_DIR = ROOT / "competitor_results"


def get_mol_descriptors(mol, missing_val=np.nan):
    """Compute all RDKit descriptors for a molecule."""
    result = {}
    for name, fn in Descriptors._descList:
        try:
            value = fn(mol)
        except Exception:
            value = missing_val
        result[name] = value
    return result


def load_first_existing(paths):
    for path in paths:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(f"None of these files exist: {paths}")


def load_models():
    """Load the best available model artifacts for this repo."""
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


def build_features(df, selected_descriptors, mol2vec_model):
    """Build aligned 2D and mol2vec features for rows with valid SMILES."""
    valid_indices = []
    x_2d_rows = []
    x_mol2vec_rows = []

    for idx, row in df.iterrows():
        api_smiles = row["API_Smiles"]
        excipient_smiles = row["Excipient_Smiles"]

        mol_api = Chem.MolFromSmiles(api_smiles)
        mol_excipient = Chem.MolFromSmiles(excipient_smiles)
        if mol_api is None or mol_excipient is None:
            logging.warning(
                "Skipping row %s due to invalid SMILES (API_CID=%s, Excipient_CID=%s)",
                idx,
                row.get("API_CID", "?"),
                row.get("Excipient_CID", "?"),
            )
            continue

        try:
            api_desc = get_mol_descriptors(mol_api)
            excipient_desc = get_mol_descriptors(mol_excipient)

            row_2d = {}
            row_2d.update(api_desc)
            row_2d.update({f"{name}_exp": value for name, value in excipient_desc.items()})
            x_2d_rows.append([row_2d.get(name, np.nan) for name in selected_descriptors])

            api_sentence = MolSentence(mol2alt_sentence(Chem.AddHs(mol_api), 1))
            excipient_sentence = MolSentence(mol2alt_sentence(Chem.AddHs(mol_excipient), 1))
            api_vec = np.asarray(sentences2vec([api_sentence], mol2vec_model, unseen="UNK")[0], dtype=float)
            excipient_vec = np.asarray(sentences2vec([excipient_sentence], mol2vec_model, unseen="UNK")[0], dtype=float)
            x_mol2vec_rows.append(np.concatenate([api_vec, excipient_vec], axis=0))
            valid_indices.append(idx)
        except Exception as exc:
            logging.warning(
                "Skipping row %s due to feature extraction error (API_CID=%s, Excipient_CID=%s): %s",
                idx,
                row.get("API_CID", "?"),
                row.get("Excipient_CID", "?"),
                exc,
            )

    x_2d = np.asarray(x_2d_rows, dtype=float)
    x_mol2vec = np.asarray(x_mol2vec_rows, dtype=float)
    return valid_indices, x_2d, x_mol2vec


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    logging.info("Loaded %d rows from %s", len(df), INPUT_PATH)

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
    logging.info("Loaded inference models from %s", COMPETITOR_DIR if COMPETITOR_DIR.exists() else ROOT)

    valid_indices, x_2d, x_mol2vec = build_features(df, selected_descriptors, mol2vec_model)
    logging.info("Feature matrix shapes - 2D: %s, mol2vec: %s", x_2d.shape, x_mol2vec.shape)

    if len(valid_indices) == 0:
        raise RuntimeError("No valid rows were found in validation_pairs.csv")

    if imputer_2d is not None:
        x_2d = imputer_2d.transform(x_2d)
        logging.info("Applied 2D imputer from competitor_results/imputer_2D.pkl")
    elif np.isnan(x_2d).any():
        logging.warning("No 2D imputer found; filling remaining NaNs with 0.0 for inference.")
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

    out_df.to_csv(OUTPUT_PATH, index=False)
    logging.info("Wrote predictions to %s", OUTPUT_PATH)
    logging.info("Valid rows predicted: %d/%d", len(valid_indices), len(df))


if __name__ == "__main__":
    main()