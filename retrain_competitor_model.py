"""
Retrain Competitor Stacking Model (Nguyen et al. 2024) on Our Split
====================================================================
Faithful reproduction of the architecture from:
  Nguyen et al., Int. J. Pharmaceutics 653 (2024) 123884
  Repo: phuongnvp/drug_exp_compatibility

Uses OUR pre-existing train/val/test split (2030/842/672).
All bugs from COMPETITOR_MODEL_FIXES.md are fixed inline.
"""

import os
import sys
import time
import json
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SVMSMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, average_precision_score, log_loss,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, cohen_kappa_score
)
import joblib
import ghostml

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "competitor_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log(msg):
    """Timestamped log."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_all_metrics(y_true, y_pred, y_prob, label=""):
    """Compute all requested metrics and return as dict."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    # Log loss (binary cross-entropy) — clip probs to avoid log(0)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    metrics['log_loss'] = log_loss(y_true, y_prob_clipped)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    if label:
        log(f"\n{'='*60}")
        log(f"  METRICS: {label}")
        log(f"{'='*60}")
        log(f"  Accuracy:   {metrics['accuracy']:.4f}")
        log(f"  Precision:  {metrics['precision']:.4f}")
        log(f"  Recall:     {metrics['recall']:.4f}")
        log(f"  F1 Score:   {metrics['f1_score']:.4f}")
        log(f"  MCC:        {metrics['mcc']:.4f}")
        log(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
        log(f"  Log Loss:   {metrics['log_loss']:.4f}")
        cm = np.array(metrics['confusion_matrix'])
        log(f"  Confusion Matrix:")
        log(f"    TN={cm[0,0]}  FP={cm[0,1]}")
        log(f"    FN={cm[1,0]}  TP={cm[1,1]}")
        log(f"{'='*60}\n")
    return metrics


def save_confusion_matrix_plot(y_true, y_pred, filename, title="Confusion Matrix"):
    """Save confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    log(f"  Saved confusion matrix plot: {filepath}")


def getMolDescriptors(mol, missingVal=None):
    """Compute all RDKit 2D descriptors for a molecule."""
    res = {}
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            val = missingVal
        res[nm] = val
    return res


# ============================================================================
# PHASE 0: LOAD DATA & VERIFY SPLIT ALIGNMENT
# ============================================================================

log("=" * 70)
log("PHASE 0: Loading datasets and verifying split alignment")
log("=" * 70)

train_df = pd.read_csv(os.path.join(BASE_DIR, 'my_dataset', 'train.csv'))
val_df = pd.read_csv(os.path.join(BASE_DIR, 'my_dataset', 'val.csv'))
test_df = pd.read_csv(os.path.join(BASE_DIR, 'my_dataset', 'test.csv'))
dataset_df = pd.read_csv(os.path.join(BASE_DIR, 'dataset.csv'))

log(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
log(f"Total: {len(train_df) + len(val_df) + len(test_df)}")
log(f"dataset.csv: {dataset_df.shape}")

log(f"Train Outcome1 distribution:\n{train_df['Outcome1'].value_counts().to_string()}")
log(f"Val Outcome1 distribution:\n{val_df['Outcome1'].value_counts().to_string()}")
log(f"Test Outcome1 distribution:\n{test_df['Outcome1'].value_counts().to_string()}")

# Verify alignment with dataset.csv
for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    merged = split_df.merge(dataset_df, on=['API_CID', 'Excipient_CID'], how='inner', suffixes=('', '_ds'))
    log(f"  {split_name}: {len(merged)}/{len(split_df)} rows matched dataset.csv by (API_CID, Excipient_CID)")

y_train = train_df['Outcome1'].values
y_val = val_df['Outcome1'].values
y_test = test_df['Outcome1'].values

# ============================================================================
# PHASE 1A: MOL2VEC FEATURE ENGINEERING
# ============================================================================

log("\n" + "=" * 70)
log("PHASE 1A: Mol2vec feature engineering")
log("=" * 70)

from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

w2vec_model = word2vec.Word2Vec.load(os.path.join(BASE_DIR, 'model_300dim.pkl'))
log(f"Loaded mol2vec model: vector_size={w2vec_model.wv.vector_size}")


def compute_mol2vec_features(df, w2vec_model, split_name):
    """Compute mol2vec features for a split dataframe."""
    log(f"  Computing mol2vec features for {split_name} ({len(df)} rows)...")

    # Parse molecules with error handling
    mol_api_list = []
    mol_exp_list = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name} mol2vec"):
        try:
            mol_api = Chem.MolFromSmiles(row['API_Smiles'])
            mol_exp = Chem.MolFromSmiles(row['Excipient_Smiles'])
            if mol_api is None or mol_exp is None:
                log(f"    WARNING: Failed to parse SMILES at row {idx} "
                    f"(API_CID={row['API_CID']}, Exc_CID={row['Excipient_CID']})")
                continue
            mol_api = Chem.AddHs(mol_api)
            mol_exp = Chem.AddHs(mol_exp)
            mol_api_list.append(mol_api)
            mol_exp_list.append(mol_exp)
            valid_indices.append(idx)
        except Exception as e:
            log(f"    ERROR at row {idx}: {e}")
            continue

    # Generate mol2vec sentences and vectors
    sentences_api = [MolSentence(mol2alt_sentence(m, 1)) for m in mol_api_list]
    sentences_exp = [MolSentence(mol2alt_sentence(m, 1)) for m in mol_exp_list]

    vecs_api = sentences2vec(sentences_api, w2vec_model, unseen='UNK')
    vecs_exp = sentences2vec(sentences_exp, w2vec_model, unseen='UNK')

    X1 = np.array([x for x in vecs_api])
    X2 = np.array([x for x in vecs_exp])

    # Log UNK rate
    vocab = set(w2vec_model.wv.key_to_index.keys())
    total_tokens_api = 0
    unk_tokens_api = 0
    for sent in sentences_api:
        for token in sent:
            total_tokens_api += 1
            if str(token) not in vocab:
                unk_tokens_api += 1
    total_tokens_exp = 0
    unk_tokens_exp = 0
    for sent in sentences_exp:
        for token in sent:
            total_tokens_exp += 1
            if str(token) not in vocab:
                unk_tokens_exp += 1

    if total_tokens_api > 0:
        log(f"    API UNK rate: {unk_tokens_api}/{total_tokens_api} = {unk_tokens_api/total_tokens_api:.4f}")
    if total_tokens_exp > 0:
        log(f"    Excipient UNK rate: {unk_tokens_exp}/{total_tokens_exp} = {unk_tokens_exp/total_tokens_exp:.4f}")

    X_mol2vec = np.concatenate([X1, X2], axis=1)
    log(f"    Mol2vec feature shape: {X_mol2vec.shape}")
    return X_mol2vec, valid_indices


X_train_mol2vec, train_valid_idx = compute_mol2vec_features(train_df, w2vec_model, "Train")
X_val_mol2vec, val_valid_idx = compute_mol2vec_features(val_df, w2vec_model, "Val")
X_test_mol2vec, test_valid_idx = compute_mol2vec_features(test_df, w2vec_model, "Test")

# Align labels with valid indices
y_train_mol2vec = train_df.loc[train_valid_idx, 'Outcome1'].values
y_val_mol2vec = val_df.loc[val_valid_idx, 'Outcome1'].values
y_test_mol2vec = test_df.loc[test_valid_idx, 'Outcome1'].values

log(f"Mol2vec features — Train: {X_train_mol2vec.shape}, Val: {X_val_mol2vec.shape}, Test: {X_test_mol2vec.shape}")

# ============================================================================
# PHASE 1B: 2D DESCRIPTOR FEATURE ENGINEERING
# ============================================================================

log("\n" + "=" * 70)
log("PHASE 1B: 2D descriptor feature engineering")
log("=" * 70)

# Load selected descriptors from variables.txt
with open(os.path.join(BASE_DIR, 'variables.txt'), 'r') as f:
    selected_descriptors = f.read().strip().split(',')
log(f"Selected descriptors from variables.txt: {len(selected_descriptors)} features")


def compute_2d_features(df, selected_descriptors, split_name):
    """Compute 2D RDKit descriptor features for a split dataframe."""
    log(f"  Computing 2D descriptors for {split_name} ({len(df)} rows)...")

    api_descs = []
    exp_descs = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name} 2D"):
        try:
            mol_api = Chem.MolFromSmiles(row['API_Smiles'])
            mol_exp = Chem.MolFromSmiles(row['Excipient_Smiles'])

            if mol_api is None or mol_exp is None:
                log(f"    WARNING: Failed to parse SMILES at row {idx}")
                continue

            desc_api = getMolDescriptors(mol_api)
            desc_exp = getMolDescriptors(mol_exp)

            api_descs.append(desc_api)
            # Add _exp suffix for excipient descriptors (matching variables.txt convention)
            exp_descs.append({k + '_exp': v for k, v in desc_exp.items()})
            valid_indices.append(idx)
        except Exception as e:
            log(f"    ERROR at row {idx}: {e}")
            continue

    df_api = pd.DataFrame(api_descs)
    df_exp = pd.DataFrame(exp_descs)
    data_2D = pd.concat([df_api, df_exp], axis=1)

    # Select only the columns from variables.txt
    missing_cols = [c for c in selected_descriptors if c not in data_2D.columns]
    if missing_cols:
        log(f"    WARNING: {len(missing_cols)} descriptor columns from variables.txt not found. "
            f"First 5: {missing_cols[:5]}")
        # Add missing columns as NaN
        for c in missing_cols:
            data_2D[c] = np.nan

    X_2D = data_2D[selected_descriptors].values.astype(float)

    # Count NaNs
    nan_total = np.isnan(X_2D).sum()
    nan_rows = np.any(np.isnan(X_2D), axis=1).sum()
    log(f"    NaN count: {nan_total} total across {nan_rows}/{len(X_2D)} rows")
    log(f"    2D feature shape: {X_2D.shape}")

    return X_2D, valid_indices


X_train_2D, train_valid_idx_2d = compute_2d_features(train_df, selected_descriptors, "Train")
X_val_2D, val_valid_idx_2d = compute_2d_features(val_df, selected_descriptors, "Val")
X_test_2D, test_valid_idx_2d = compute_2d_features(test_df, selected_descriptors, "Test")

# Align labels
y_train_2D = train_df.loc[train_valid_idx_2d, 'Outcome1'].values
y_val_2D = val_df.loc[val_valid_idx_2d, 'Outcome1'].values
y_test_2D = test_df.loc[test_valid_idx_2d, 'Outcome1'].values

# Impute NaNs — fit on train, transform all
log("  Fitting imputer on training 2D features...")
imputer = SimpleImputer(strategy='median')
X_train_2D = imputer.fit_transform(X_train_2D)
X_val_2D = imputer.transform(X_val_2D)
X_test_2D = imputer.transform(X_test_2D)
log(f"  Post-imputation NaN check — Train: {np.isnan(X_train_2D).sum()}, "
    f"Val: {np.isnan(X_val_2D).sum()}, Test: {np.isnan(X_test_2D).sum()}")

log(f"2D features — Train: {X_train_2D.shape}, Val: {X_val_2D.shape}, Test: {X_test_2D.shape}")

# ============================================================================
# PHASE 2: MODEL SELECTION — 8-ALGORITHM GRID SEARCH (BOTH BRANCHES)
# ============================================================================

log("\n" + "=" * 70)
log("PHASE 2: 8-algorithm grid search on both feature branches")
log("=" * 70)

# SVMSMOTE on training data — Mol2vec branch
log("  Applying SVMSMOTE to mol2vec training data...")
svmsmote_mol = SVMSMOTE(random_state=SEED)
X_train_mol2vec_res, y_train_mol2vec_res = svmsmote_mol.fit_resample(X_train_mol2vec, y_train_mol2vec)
log(f"  Post-SMOTE mol2vec: {X_train_mol2vec_res.shape}, class dist: "
    f"{np.bincount(y_train_mol2vec_res)}")

# SVMSMOTE on training data — 2D branch
log("  Applying SVMSMOTE to 2D training data...")
svmsmote_2d = SVMSMOTE(random_state=SEED)
X_train_2D_res, y_train_2D_res = svmsmote_2d.fit_resample(X_train_2D, y_train_2D)
log(f"  Post-SMOTE 2D: {X_train_2D_res.shape}, class dist: "
    f"{np.bincount(y_train_2D_res)}")

# Define grids using the already found optimal hyperparameters to fast-forward execution
best_params_mol2vec = {
    'RandomForest': {'n_estimators': [200]},
    'DecisionTree': {'max_depth': [25]},
    'SVM': {'C': [10], 'gamma': [0.001]},
    'MLP': {
        'activation': ['relu'],
        'hidden_layer_sizes': [(500,)],
        'learning_rate': ['adaptive'],
        'momentum': [0.1],
        'solver': ['adam'],
    },
    'kNN': {'n_neighbors': [4]},
    'LogisticRegression': {'C': [10]},
    'XGBoost': {
        'n_estimators': [400],
        'max_depth': [15],
        'learning_rate': [0.5],
    },
    'AdaBoost': {
        'n_estimators': [700],
        'learning_rate': [0.9],
    },
}

best_params_2D = {
    'RandomForest': {'n_estimators': [200]},
    'DecisionTree': {'max_depth': [15]},
    'SVM': {'C': [5], 'gamma': [0.001]},
    'MLP': {
        'activation': ['relu'],
        'hidden_layer_sizes': [(100,)],
        'learning_rate': ['adaptive'],
        'momentum': [0.1],
        'solver': ['adam'],
    },
    'kNN': {'n_neighbors': [4]},
    'LogisticRegression': {'C': [0.1]},
    'XGBoost': {
        'n_estimators': [400],
        'max_depth': [5],
        'learning_rate': [0.7],
    },
    'AdaBoost': {
        'n_estimators': [400],
        'learning_rate': [0.9],
    },
}

algorithms = {
    'RandomForest': {'estimator': RandomForestClassifier(random_state=SEED)},
    'DecisionTree': {'estimator': DecisionTreeClassifier(random_state=SEED)},
    'SVM': {'estimator': SVC(random_state=SEED, probability=True)},
    'MLP': {'estimator': MLPClassifier(random_state=SEED)},
    'kNN': {'estimator': KNeighborsClassifier()},
    'LogisticRegression': {'estimator': LogisticRegression(random_state=SEED, max_iter=1000)},
    'XGBoost': {'estimator': XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss')},
    'AdaBoost': {'estimator': AdaBoostClassifier(random_state=SEED, algorithm='SAMME')},
}

grid_search_results = {'mol2vec': {}, '2D': {}}


def run_grid_search(X_train, y_train, X_val, y_val, branch_name, best_grids):
    """Run 8-algorithm grid search on a feature branch (fast-forwarded)."""
    results = {}
    for i, (name, config) in enumerate(algorithms.items(), 1):
        log(f"  [{i}/8] Grid search: {name} ({branch_name} features)...")
        t0 = time.time()

        param_grid = best_grids[name]
        gs = GridSearchCV(
            config['estimator'],
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
            verbose=0
        )
        gs.fit(X_train, y_train)

        y_pred = gs.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred)
        elapsed = time.time() - t0

        results[name] = {
            'best_params': gs.best_params_,
            'best_cv_score': gs.best_score_,
            'val_accuracy': val_acc,
            'grid_search_obj': gs,
            'time': elapsed,
        }
        log(f"         done in {elapsed:.1f}s — best CV acc {gs.best_score_:.4f}, "
            f"val acc {val_acc:.4f}, params: {gs.best_params_}")

    return results


log("\n--- Mol2vec Branch Grid Search ---")
grid_search_results['mol2vec'] = run_grid_search(
    X_train_mol2vec_res, y_train_mol2vec_res, X_val_mol2vec, y_val_mol2vec, "mol2vec", best_params_mol2vec
)

log("\n--- 2D Descriptor Branch Grid Search ---")
grid_search_results['2D'] = run_grid_search(
    X_train_2D_res, y_train_2D_res, X_val_2D, y_val_2D, "2D", best_params_2D
)

# Print grid search summary
for branch in ['mol2vec', '2D']:
    log(f"\n--- {branch} Branch Grid Search Summary ---")
    sorted_algos = sorted(grid_search_results[branch].items(), key=lambda x: x[1]['val_accuracy'], reverse=True)
    for rank, (name, res) in enumerate(sorted_algos, 1):
        log(f"  #{rank} {name}: val_acc={res['val_accuracy']:.4f} (CV={res['best_cv_score']:.4f})")

# ============================================================================
# PHASE 3: STACKING MODELS (BOTH BRANCHES)
# ============================================================================

log("\n" + "=" * 70)
log("PHASE 3: Building stacking models (AdaBoost + RF + XGBoost -> LR)")
log("=" * 70)

# Use grid-search-found best params for the 3 stacking base models
# But force the SAME 3 algorithms as the paper: AdaBoost, RandomForest, XGBoost


def build_and_train_stack(X_train, y_train, X_val, y_val, gs_results, branch_name):
    """Build and train a stacking classifier on a feature branch."""
    ada_params = gs_results['AdaBoost']['best_params']
    rf_params = gs_results['RandomForest']['best_params']
    xgb_params = gs_results['XGBoost']['best_params']

    log(f"  {branch_name} - Using grid-searched params:")
    log(f"    AdaBoost: {ada_params}")
    log(f"    RandomForest: {rf_params}")
    log(f"    XGBoost: {xgb_params}")

    model1 = AdaBoostClassifier(random_state=SEED, algorithm='SAMME', **ada_params)
    model2 = RandomForestClassifier(random_state=SEED, **rf_params)
    model3 = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss', **xgb_params)

    lr_meta = LogisticRegression(C=10, random_state=SEED, max_iter=1000)
    stack_model = StackingClassifier(
        estimators=[('ada', model1), ('rf', model2), ('xgb', model3)],
        final_estimator=lr_meta,
        cv=5,
        passthrough=False
    )

    # Apply class weights via sample_weight (paper: {0: 0.3, 1: 0.7})
    class_weights = {0: 0.3, 1: 0.7}
    sample_weights = np.array([class_weights[i] for i in y_train])

    log(f"  Training {branch_name} stacking model...")
    t0 = time.time()
    stack_model.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time.time() - t0
    log(f"  {branch_name} stacking model trained in {elapsed:.1f}s")

    # Evaluate on validation set
    y_pred = stack_model.predict(X_val)
    y_prob = stack_model.predict_proba(X_val)[:, 1]
    log(f"  {branch_name} Val Accuracy: {accuracy_score(y_val, y_pred):.4f}, "
        f"F1: {f1_score(y_val, y_pred):.4f}")

    return stack_model


log("\n--- Mol2vec Stacking Model ---")
stack_mol2vec = build_and_train_stack(
    X_train_mol2vec_res, y_train_mol2vec_res,
    X_val_mol2vec, y_val_mol2vec,
    grid_search_results['mol2vec'], "mol2vec"
)

log("\n--- 2D Stacking Model ---")
stack_2D = build_and_train_stack(
    X_train_2D_res, y_train_2D_res,
    X_val_2D, y_val_2D,
    grid_search_results['2D'], "2D"
)

# Save individual branch models
joblib.dump(stack_mol2vec, os.path.join(OUTPUT_DIR, 'model_mol2vec_retrained.pkl'))
joblib.dump(stack_2D, os.path.join(OUTPUT_DIR, 'model_2D_retrained.pkl'))
log("  Saved retrained branch models to competitor_results/")

# ============================================================================
# PHASE 4: COMBINED META-LEARNER + THRESHOLD TUNING
# ============================================================================

log("\n" + "=" * 70)
log("PHASE 4: Combined meta-learner (LR on branch probabilities)")
log("=" * 70)

# --- Align indices for combined model ---
# We need rows that are valid in BOTH mol2vec and 2D branches
train_common_idx = sorted(set(train_valid_idx) & set(train_valid_idx_2d))
val_common_idx = sorted(set(val_valid_idx) & set(val_valid_idx_2d))
test_common_idx = sorted(set(test_valid_idx) & set(test_valid_idx_2d))

log(f"  Common valid indices — Train: {len(train_common_idx)}, Val: {len(val_common_idx)}, Test: {len(test_common_idx)}")

# Map from valid_idx lists to positional indices in feature arrays
def idx_to_pos(common, valid_list):
    """Map common indices to positional indices in the feature array."""
    valid_to_pos = {idx: pos for pos, idx in enumerate(valid_list)}
    return [valid_to_pos[idx] for idx in common]

train_pos_mol = idx_to_pos(train_common_idx, train_valid_idx)
train_pos_2d = idx_to_pos(train_common_idx, train_valid_idx_2d)
val_pos_mol = idx_to_pos(val_common_idx, val_valid_idx)
val_pos_2d = idx_to_pos(val_common_idx, val_valid_idx_2d)
test_pos_mol = idx_to_pos(test_common_idx, test_valid_idx)
test_pos_2d = idx_to_pos(test_common_idx, test_valid_idx_2d)

X_train_mol_common = X_train_mol2vec[train_pos_mol]
X_train_2d_common = X_train_2D[train_pos_2d]
y_train_common = train_df.loc[train_common_idx, 'Outcome1'].values

log("  Applying SVMSMOTE to common training data for combined model...")
svmsmote_combined = SVMSMOTE(random_state=SEED)
X_train_combined_raw = np.concatenate([X_train_mol_common, X_train_2d_common], axis=1)
X_train_combined_res, y_train_combined_res = svmsmote_combined.fit_resample(X_train_combined_raw, y_train_common)

n_mol = X_train_mol_common.shape[1]
X_train_mol_res_c = X_train_combined_res[:, :n_mol]
X_train_2d_res_c = X_train_combined_res[:, n_mol:]
log(f"  Post-SMOTE combined: {X_train_combined_res.shape}, class dist: {np.bincount(y_train_combined_res)}")

y_train_prob_mol = stack_mol2vec.predict_proba(X_train_mol_res_c)[:, 1]
y_train_prob_2d = stack_2D.predict_proba(X_train_2d_res_c)[:, 1]
X_meta_train = np.stack([y_train_prob_2d, y_train_prob_mol], axis=1)

lr_combined = LogisticRegression(random_state=SEED, max_iter=1000)
lr_combined.fit(X_meta_train, y_train_combined_res)
log(f"  Combined meta-learner trained. Coefficients: {lr_combined.coef_}, Intercept: {lr_combined.intercept_}")

X_val_mol_common = X_val_mol2vec[val_pos_mol]
X_val_2d_common = X_val_2D[val_pos_2d]
y_val_common = val_df.loc[val_common_idx, 'Outcome1'].values

y_val_prob_mol = stack_mol2vec.predict_proba(X_val_mol_common)[:, 1]
y_val_prob_2d = stack_2D.predict_proba(X_val_2d_common)[:, 1]
X_meta_val = np.stack([y_val_prob_2d, y_val_prob_mol], axis=1)

y_val_pred_combined = lr_combined.predict(X_meta_val)
y_val_prob_combined = lr_combined.predict_proba(X_meta_val)[:, 1]

log(f"  Combined model (default threshold 0.5) — Val Accuracy: {accuracy_score(y_val_common, y_val_pred_combined):.4f}, "
    f"F1: {f1_score(y_val_common, y_val_pred_combined):.4f}")

log("\n  Searching for F1-optimal threshold on validation set...")
thresholds = np.arange(0, 1, 0.001)
f1_scores_thresh = [f1_score(y_val_common, (y_val_prob_combined >= t).astype(int), zero_division=0)
                    for t in thresholds]
best_f1_idx = np.argmax(f1_scores_thresh)
threshold_f1 = thresholds[best_f1_idx]
log(f"  F1-optimal threshold: {threshold_f1:.3f}, F1={f1_scores_thresh[best_f1_idx]:.5f}")

log("\n  Computing GHOST threshold (Kappa-based)...")
train_probs_combined = lr_combined.predict_proba(X_meta_train)[:, 1]
ghost_thresholds = np.round(np.arange(0, 1, 0.001), 3)
try:
    threshold_ghost = ghostml.optimize_threshold_from_predictions(
        y_train_combined_res, train_probs_combined, ghost_thresholds, ThOpt_metrics='Kappa'
    )
    log(f"  GHOST threshold: {threshold_ghost:.3f}")
except Exception as e:
    log(f"  WARNING: GHOST threshold optimization failed: {e}")
    threshold_ghost = 0.5

joblib.dump(lr_combined, os.path.join(OUTPUT_DIR, 'model_lr_retrained.pkl'))
joblib.dump(imputer, os.path.join(OUTPUT_DIR, 'imputer_2D.pkl'))

# ============================================================================
# PHASE 5: COMPREHENSIVE EVALUATION ON TEST SET
# ============================================================================

log("\n" + "=" * 70)
log("PHASE 5: Comprehensive evaluation on TEST set")
log("=" * 70)

X_test_mol_common = X_test_mol2vec[test_pos_mol]
X_test_2d_common = X_test_2D[test_pos_2d]
y_test_common = test_df.loc[test_common_idx, 'Outcome1'].values

all_results = {}

# Mol2vec branch alone
y_test_prob_mol = stack_mol2vec.predict_proba(X_test_mol_common)[:, 1]
y_test_pred_mol = (y_test_prob_mol >= 0.5).astype(int)
metrics_mol = compute_all_metrics(y_test_common, y_test_pred_mol, y_test_prob_mol,
                                   "Mol2vec Branch (test, threshold=0.5)")
metrics_mol['threshold'] = 0.5
all_results['mol2vec_branch'] = metrics_mol
save_confusion_matrix_plot(y_test_common, y_test_pred_mol,
                           "cm_mol2vec_branch.png", "Mol2vec Branch - Test Set")

# 2D branch alone
y_test_prob_2d = stack_2D.predict_proba(X_test_2d_common)[:, 1]
y_test_pred_2d = (y_test_prob_2d >= 0.5).astype(int)
metrics_2d = compute_all_metrics(y_test_common, y_test_pred_2d, y_test_prob_2d,
                                  "2D Descriptor Branch (test, threshold=0.5)")
metrics_2d['threshold'] = 0.5
all_results['2D_branch'] = metrics_2d
save_confusion_matrix_plot(y_test_common, y_test_pred_2d,
                           "cm_2D_branch.png", "2D Branch - Test Set")

# Combined model evaluation
X_meta_test = np.stack([y_test_prob_2d, y_test_prob_mol], axis=1)
y_test_prob_combined = lr_combined.predict_proba(X_meta_test)[:, 1]

# Default threshold (0.5)
y_test_pred_default = lr_combined.predict(X_meta_test)
metrics_default = compute_all_metrics(y_test_common, y_test_pred_default, y_test_prob_combined,
                                       "Combined Model (test, threshold=0.5)")
metrics_default['threshold'] = 0.5
all_results['combined_default'] = metrics_default
save_confusion_matrix_plot(y_test_common, y_test_pred_default,
                           "cm_combined_default.png", "Combined Model (threshold=0.5) - Test Set")

# F1-optimized threshold
y_test_pred_f1 = (y_test_prob_combined >= threshold_f1).astype(int)
metrics_f1 = compute_all_metrics(y_test_common, y_test_pred_f1, y_test_prob_combined,
                                  f"Combined Model (test, F1-threshold={threshold_f1:.3f})")
metrics_f1['threshold'] = float(threshold_f1)
all_results['combined_f1_threshold'] = metrics_f1
save_confusion_matrix_plot(y_test_common, y_test_pred_f1,
                           "cm_combined_f1_threshold.png",
                           f"Combined Model (F1-threshold={threshold_f1:.3f}) - Test Set")

# GHOST threshold
y_test_pred_ghost = (y_test_prob_combined >= threshold_ghost).astype(int)
metrics_ghost = compute_all_metrics(y_test_common, y_test_pred_ghost, y_test_prob_combined,
                                     f"Combined Model (test, GHOST-threshold={threshold_ghost:.3f})")
metrics_ghost['threshold'] = float(threshold_ghost)
all_results['combined_ghost_threshold'] = metrics_ghost
save_confusion_matrix_plot(y_test_common, y_test_pred_ghost,
                           "cm_combined_ghost_threshold.png",
                           f"Combined Model (GHOST-threshold={threshold_ghost:.3f}) - Test Set")

# ============================================================================
# SAVE ALL RESULTS
# ============================================================================

# Save to JSON
results_file = os.path.join(OUTPUT_DIR, 'competitor_results.json')
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)
log(f"\nAll results saved to: {results_file}")

# ============================================================================
# FINAL SUMMARY TABLE
# ============================================================================

log("\n" + "=" * 70)
log("FINAL SUMMARY TABLE - Competitor Model on Our Split (Test Set)")
log("=" * 70)

header = f"{'Configuration':<40} {'Thresh':>6} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'MCC':>7} {'PR-AUC':>7} {'Loss':>7}"
log(header)
log("-" * len(header))

for config_name, metrics in all_results.items():
    display_name = config_name.replace('_', ' ').title()
    log(f"{display_name:<40} {metrics['threshold']:>6.3f} "
        f"{metrics['accuracy']:>7.4f} {metrics['precision']:>7.4f} "
        f"{metrics['recall']:>7.4f} {metrics['f1_score']:>7.4f} "
        f"{metrics['mcc']:>7.4f} {metrics['pr_auc']:>7.4f} "
        f"{metrics['log_loss']:>7.4f}")

log("\n" + "=" * 70)
log("DONE. All models retrained, metrics computed, plots saved.")
log("=" * 70)
