"""
Competitor model (Nguyen et al., Int. J. Pharm. 653 (2024) 123884) on the 80/20 split
=====================================================================================
Same pipeline as retrain_competitor_model.py (the 60/20/20 run), trained from scratch on the
split in --data-dir (default: paper_model/new_data/):

  train : <data-dir>/train.csv                   -> training (SVM-SMOTE, grid search, stacks)
  val   : <data-dir>/val.csv                     -> model selection + threshold tuning only
  test  : <data-dir>/held_out_test_set.csv       -> held-out pairs, scored only, never tuned on
          (test.csv or heldout.csv are also accepted)
The label column may be named Outcome1, ground_truth or "Output Value".

Differences vs retrain_competitor_model.py
  * The FULL paper hyperparameter grids are searched (no fast-forwarded single-value grids).
  * Class-weight ablation of the paper (Table 3): 0.1:0.9 ... 0.5:0.5 for both branch stacks,
    and a combined model for every class weight. The "main" model is 0.3:0.7 (paper's choice).
  * Every model is scored on the 24 pairs: 8 tuned algorithms x 2 branches, the branch stacks,
    and the combined models.
  * Threshold ablations for every model: 0.5 default, F1-optimal on val (paper method), the
    paper's published thresholds 0.202 / 0.020. Combined models also get GHOST (on train
    probabilities as in the paper's code, and on val probabilities), plus a full 0..1 sweep.
  * Grid-search and stack fits are checkpointed in <out-dir>; re-running resumes.

Run
  python train_80_20_and_infer_24.py                 # full run on new_data/ (~1.5-2.5 h)
  python train_80_20_and_infer_24.py --smoke         # 1-3 min code check on a subsample
  python train_80_20_and_infer_24.py --drop-test-overlap   # remove test pairs found in train
  python train_80_20_and_infer_24.py --data-dir ../project/metrics/pgb_fixed_vector_morgan_concat_asl/single_run_80_20split
                                                     # the earlier split (resumes results_80_20split/)

Outputs (in --out-dir, default results_<data-dir name>/, e.g. results_new_data/)
  predictions_24_wide.csv     one row per pair, probability from every model
  predictions_24_long.csv     model x threshold rule x pair -> prob, pred, correct
  metrics_all.csv             model x threshold rule x split(val/test) -> all metrics
  summary_24_pairs.csv        test-only view of metrics_all.csv, one row per model/rule
  threshold_sweep.csv         every model, thresholds 0.000..1.000 step 0.001, val and test
  grid_search_results.csv     full cv_results_ of every grid search
  run_info.json               args, data checks, package versions, timings
  features/, models/, grid_search/   cached features and fitted models
"""

import os
# Must be set before joblib / sklearn / imblearn are imported:
#  * LOKY_MAX_CPU_COUNT below the logical core count makes loky skip its physical-core probe,
#    which shells out to `wmic` (removed in Windows 11) and prints a traceback.
#  * PYTHONWARNINGS is inherited by joblib worker processes; warnings.filterwarnings() below
#    only affects this process, so ConvergenceWarnings from grid-search workers leaked through.
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(max(1, (os.cpu_count() or 2) - 1)))
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

import argparse
import hashlib
import json
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')

import sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SVMSMOTE
from xgboost import XGBClassifier
import xgboost

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'new_data')
# Earlier split: its results live in results_80_20split/ (kept so that run can still resume)
OLD_SPLIT_DIR = os.path.join(
    os.path.dirname(BASE_DIR), 'project', 'metrics',
    'pgb_fixed_vector_morgan_concat_asl', 'single_run_80_20split',
)
TEST_FILE_NAMES = ['held_out_test_set.csv', 'test.csv', 'heldout.csv']
LABEL_COLUMNS = ['Outcome1', 'ground_truth', 'Output Value']

# Paper's published thresholds (Table 5): F1-based and GHOST
PAPER_F1_THRESHOLD = 0.202
PAPER_GHOST_THRESHOLD = 0.020

# Grids exactly as in the authors' model_selection.py
PAPER_GRIDS = {
    'RandomForest': {'n_estimators': [50, 100, 150, 200, 300]},
    'DecisionTree': {'max_depth': [5, 15, 25, 30, 40, 50]},
    'SVM': {'gamma': [0.001, 0.01, 0.05, 0.1], 'C': [1, 5, 10, 100]},
    'MLP': {
        'hidden_layer_sizes': [(100,), (300,), (500,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'momentum': [0.1, 0.5, 0.9],
        'learning_rate': ['adaptive'],
    },
    'kNN': {'n_neighbors': list(range(3, 16))},
    'LogisticRegression': {'C': [0.01, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 0.8, 1, 10, 15]},
    'XGBoost': {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [3, 5, 7, 9, 15],
        'learning_rate': [0.3, 0.5, 0.7, 0.9],
    },
    'AdaBoost': {'n_estimators': [300, 400, 500, 600, 700], 'learning_rate': [0.5, 0.7, 0.9]},
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log(msg):
    """Timestamped log."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fingerprint(*parts):
    """Hash of arrays / objects, used to make sure a checkpoint matches the current inputs."""
    h = hashlib.md5()
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else repr(p).encode())
    return h.hexdigest()


def make_adaboost(seed, **params):
    # 'algorithm' is deprecated from sklearn 1.6 (SAMME becomes the only option)
    if tuple(int(x) for x in sklearn.__version__.split('.')[:2]) < (1, 6):
        params.setdefault('algorithm', 'SAMME')
    return AdaBoostClassifier(random_state=seed, **params)


def make_estimator(name, seed):
    return {
        'RandomForest': lambda: RandomForestClassifier(random_state=seed),
        'DecisionTree': lambda: DecisionTreeClassifier(random_state=seed),
        # Searched without probability=True (as the paper); refit with it afterwards for predict_proba
        'SVM': lambda: SVC(random_state=seed),
        'MLP': lambda: MLPClassifier(random_state=seed),
        'kNN': lambda: KNeighborsClassifier(),
        'LogisticRegression': lambda: LogisticRegression(random_state=seed, max_iter=1000),
        'XGBoost': lambda: XGBClassifier(random_state=seed, eval_metric='logloss', n_jobs=1),
        'AdaBoost': lambda: make_adaboost(seed),
    }[name]()


def compute_metrics(y_true, y_prob, threshold):
    """All metrics at one threshold. roc_auc_hard = the paper's 'AUC' (computed on 0/1 labels)."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y_true)) == 2
    return {
        'threshold': float(threshold),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'pr_auc': average_precision_score(y_true, y_prob) if two_classes else np.nan,
        'roc_auc': roc_auc_score(y_true, y_prob) if two_classes else np.nan,
        'roc_auc_hard': roc_auc_score(y_true, y_pred) if two_classes else np.nan,
        'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp),
        'incompatible_found': f"{tp}/{tp + fn}",
        'compatible_correct': f"{tn}/{tn + fp}",
    }


def f1_optimal_threshold(y_true, y_prob):
    """Paper method: thresholds 0..1 step 0.001, argmax F1 (first maximum, as np.argmax)."""
    thresholds = np.arange(0, 1, 0.001)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    return float(round(thresholds[int(np.argmax(scores))], 3))


def ghost_threshold(y_true, y_prob, seed, n_subsets):
    """GHOST (Esposito et al. 2021) with the paper's settings: Kappa, thresholds rounded to 2 dp."""
    import ghostml
    thresholds = np.unique(np.round(np.arange(0, 1, 0.001), 2))
    return float(ghostml.optimize_threshold_from_predictions(
        y_true, y_prob, thresholds, ThOpt_metrics='Kappa', N_subsets=n_subsets, random_seed=seed,
    ))


def getMolDescriptors(mol, missingVal=None):
    """Compute all RDKit 2D descriptors for a molecule."""
    res = {}
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except Exception:
            val = missingVal
        res[nm] = val
    return res


# ============================================================================
# DATA + FEATURES
# ============================================================================

def load_split(path, name):
    df = pd.read_csv(path)
    # Held-out files name the label 'ground_truth'; use one name (Outcome1) internally
    label_col = next((c for c in LABEL_COLUMNS if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"{name} ({path}) has no label column (expected one of {LABEL_COLUMNS})")
    if label_col != 'Outcome1':
        df = df.rename(columns={label_col: 'Outcome1'})
    required = ['API_CID', 'Excipient_CID', 'Outcome1', 'API_Smiles', 'Excipient_Smiles']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} ({path}) is missing columns {missing}")
    df = df.reset_index(drop=True)
    log(f"  {name}: {os.path.basename(path)} {df.shape}, label '{label_col}' = "
        f"{df['Outcome1'].value_counts().sort_index().to_dict()}")
    return df


def compute_features(df, w2vec_model, selected_descriptors, split_name):
    """Mol2vec (API + excipient, 200) and named 2D descriptors (398) for rows that parse."""
    from mol2vec.features import MolSentence, mol2alt_sentence, sentences2vec

    vocab = set(w2vec_model.wv.key_to_index.keys())
    x_mol, x_2d, valid, failed = [], [], [], []
    unk = {'API': [0, 0], 'Excipient': [0, 0]}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name} features"):
        mol_api = Chem.MolFromSmiles(str(row['API_Smiles']))
        mol_exp = Chem.MolFromSmiles(str(row['Excipient_Smiles']))
        if mol_api is None or mol_exp is None:
            failed.append(idx)
            continue

        # 2D descriptors selected by name (variables.txt), excipient columns suffixed '_exp'
        desc = getMolDescriptors(mol_api)
        desc.update({k + '_exp': v for k, v in getMolDescriptors(mol_exp).items()})
        x_2d.append([desc.get(c, np.nan) for c in selected_descriptors])

        # Mol2vec with explicit hydrogens, radius 1, as in the authors' code
        vecs = []
        for side, mol in [('API', mol_api), ('Excipient', mol_exp)]:
            sentence = MolSentence(mol2alt_sentence(Chem.AddHs(mol), 1))
            unk[side][0] += len(sentence)
            unk[side][1] += sum(str(t) not in vocab for t in sentence)
            vecs.append(np.asarray(sentences2vec([sentence], w2vec_model, unseen='UNK')[0], dtype=float))
        x_mol.append(np.concatenate(vecs))
        valid.append(idx)

    if failed:
        log(f"    WARNING: {len(failed)} {split_name} rows failed SMILES parsing and were skipped: "
            f"{df.loc[failed, ['API_CID', 'Excipient_CID']].values.tolist()[:10]}")
    for side, (total, n_unk) in unk.items():
        if total:
            log(f"    {side} UNK rate: {n_unk}/{total} = {n_unk / total:.4f}")

    x_2d = np.asarray(x_2d, dtype=float)
    log(f"    {split_name}: mol2vec {np.asarray(x_mol).shape}, 2D {x_2d.shape}, "
        f"2D NaNs {int(np.isnan(x_2d).sum())} in {int(np.isnan(x_2d).any(axis=1).sum())} rows")
    return np.asarray(x_mol, dtype=float), x_2d, valid


def load_or_compute_features(df, split_name, cache_dir, w2vec_model, selected_descriptors):
    """Features are cached per split so later runs use identical inputs (RDKit-version drift)."""
    key = fingerprint(df[['API_Smiles', 'Excipient_Smiles']].values.astype(str), selected_descriptors)
    path = os.path.join(cache_dir, f"{split_name}_{key[:12]}.npz")
    if os.path.exists(path):
        d = np.load(path)
        log(f"  {split_name}: loaded cached features {os.path.basename(path)}")
        return d['x_mol'], d['x_2d'], d['valid'].tolist()
    x_mol, x_2d, valid = compute_features(df, w2vec_model, selected_descriptors, split_name)
    np.savez_compressed(path, x_mol=x_mol, x_2d=x_2d, valid=np.asarray(valid))
    return x_mol, x_2d, valid


# ============================================================================
# TRAINING
# ============================================================================

def run_grid_search(name, grid, X, y, branch, seed, n_jobs, ckpt_dir):
    """GridSearchCV (accuracy, 5-fold) on SMOTE-resampled train, as in the paper. Checkpointed."""
    key = fingerprint(X, y, grid, name, seed)
    path = os.path.join(ckpt_dir, f"{branch}__{name}.joblib")
    if os.path.exists(path):
        ckpt = joblib.load(path)
        if ckpt.get('fingerprint') == key:
            log(f"    {name}: resumed from checkpoint, params {ckpt['best_params']}")
            return ckpt
        log(f"    {name}: checkpoint is stale (inputs changed) - re-running")

    t0 = time.time()
    gs = GridSearchCV(make_estimator(name, seed), param_grid=grid, scoring='accuracy',
                      cv=5, n_jobs=n_jobs, verbose=1)
    gs.fit(X, y)
    model = gs.best_estimator_
    if name == 'SVM':
        model = SVC(random_state=seed, probability=True, **gs.best_params_).fit(X, y)

    ckpt = {
        'fingerprint': key,
        'best_params': gs.best_params_,
        'best_cv_accuracy': float(gs.best_score_),
        'cv_results': pd.DataFrame(gs.cv_results_).drop(columns=['params']),
        'model': model,
        'seconds': time.time() - t0,
    }
    joblib.dump(ckpt, path)
    log(f"    {name}: done in {ckpt['seconds']:.1f}s, best CV acc {gs.best_score_:.4f}, params {gs.best_params_}")
    return ckpt


def train_stack(X, y, gs_results, class_weight, branch, seed, n_jobs, ckpt_dir):
    """AdaBoost + RandomForest + XGBoost -> LogisticRegression(C=10), class weights via sample_weight."""
    ada_p = gs_results['AdaBoost']['best_params']
    rf_p = gs_results['RandomForest']['best_params']
    xgb_p = gs_results['XGBoost']['best_params']
    key = fingerprint(X, y, ada_p, rf_p, xgb_p, class_weight, seed)
    path = os.path.join(ckpt_dir, f"{branch}__stack_cw{class_weight[0]:.1f}-{class_weight[1]:.1f}.joblib")
    if os.path.exists(path):
        ckpt = joblib.load(path)
        if ckpt.get('fingerprint') == key:
            log(f"    {branch} stack cw={class_weight}: resumed from checkpoint")
            return ckpt['model']

    stack = StackingClassifier(
        estimators=[
            ('ada', make_adaboost(seed, **ada_p)),
            ('rf', RandomForestClassifier(random_state=seed, **rf_p)),
            ('xgb', XGBClassifier(random_state=seed, eval_metric='logloss', **xgb_p)),
        ],
        final_estimator=LogisticRegression(C=10, random_state=seed, max_iter=1000),
        cv=5, passthrough=False, n_jobs=n_jobs,
    )
    weights = np.where(y == 1, class_weight[1], class_weight[0])
    t0 = time.time()
    stack.fit(X, y, sample_weight=weights)
    joblib.dump({'fingerprint': key, 'model': stack}, path)
    log(f"    {branch} stack cw={class_weight}: trained in {time.time() - t0:.1f}s")
    return stack


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default=DATA_DIR,
                   help='folder with train.csv, val.csv and the held-out file (default: new_data/)')
    p.add_argument('--train', help='override: train CSV (default <data-dir>/train.csv)')
    p.add_argument('--val', help='override: val CSV (default <data-dir>/val.csv)')
    p.add_argument('--test', help=f'override: held-out CSV (default first of {TEST_FILE_NAMES} in <data-dir>)')
    p.add_argument('--out-dir', help='default results_<data-dir name>/ next to this script')
    p.add_argument('--class-weights', default='0.1:0.9,0.2:0.8,0.3:0.7,0.4:0.6,0.5:0.5',
                   help='class-weight ablation, "w0:w1" comma separated (paper Table 3)')
    p.add_argument('--main-class-weight', default='0.3:0.7', help="paper's chosen weighting")
    p.add_argument('--drop-test-overlap', action='store_true',
                   help='remove test pairs that also appear in train/val (default: keep, logged)')
    p.add_argument('--ghost-subsets', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    # Each joblib worker is a separate process with its own copy of the data and models;
    # 27 workers on a 16 GB machine exhausted Windows commit memory (WinError 1450).
    p.add_argument('--n-jobs', type=int, default=6,
                   help='parallel worker processes (default 6; -1 = all cores, needs lots of RAM)')
    p.add_argument('--smoke', action='store_true',
                   help='quick code check: subsample data, one value per grid, one class weight')
    args = p.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    args.train = args.train or os.path.join(data_dir, 'train.csv')
    args.val = args.val or os.path.join(data_dir, 'val.csv')
    if not args.test:
        found = [os.path.join(data_dir, n) for n in TEST_FILE_NAMES if os.path.exists(os.path.join(data_dir, n))]
        if not found:
            p.error(f"no held-out file in {data_dir} (looked for {TEST_FILE_NAMES}); pass --test")
        args.test = found[0]
    for path in [args.train, args.val, args.test]:
        if not os.path.exists(path):
            p.error(f"file not found: {path}")
    if not args.out_dir:
        same_as_old = os.path.normcase(data_dir) == os.path.normcase(os.path.abspath(OLD_SPLIT_DIR))
        args.out_dir = os.path.join(
            BASE_DIR, 'results_80_20split' if same_as_old else f"results_{os.path.basename(data_dir)}")
    return args


def parse_cw(s):
    w0, w1 = (float(x) for x in s.split(':'))
    return (w0, w1)


def main():
    args = parse_args()
    seed = args.seed
    np.random.seed(seed)
    class_weights = [parse_cw(s) for s in args.class_weights.split(',')]
    main_cw = parse_cw(args.main_class_weight)
    if main_cw not in class_weights:
        class_weights.append(main_cw)

    out_dir = args.out_dir + ('_smoke' if args.smoke else '')
    dirs = {k: os.path.join(out_dir, k) for k in ['features', 'models', 'grid_search']}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    run_info = {'args': vars(args), 'timings_s': {}, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    t_start = time.time()

    # ------------------------------------------------------------------ PHASE 0
    log("=" * 70)
    log("PHASE 0: Loading data")
    log("=" * 70)
    log(f"  Data dir: {os.path.abspath(args.data_dir)}")
    log(f"  Output dir: {out_dir}")
    train_df = load_split(args.train, 'Train')
    val_df = load_split(args.val, 'Val')
    test_df = load_split(args.test, 'Test')

    key = lambda d: set(zip(d['API_CID'], d['Excipient_CID']))
    overlap = {
        'train_val_pairs': len(key(train_df) & key(val_df)),
        'test_in_train_pairs': sorted(map(list, key(test_df) & key(train_df))),
        'test_in_val_pairs': sorted(map(list, key(test_df) & key(val_df))),
        'test_apis_in_train': len(set(test_df['API_CID']) & set(train_df['API_CID'])),
    }
    log(f"  Overlap checks: {overlap}")
    # Flag per test pair, from the full train file (before any dropping / smoke subsampling)
    in_train = [k in key(train_df) and not args.drop_test_overlap
                for k in zip(test_df['API_CID'], test_df['Excipient_CID'])]
    if overlap['test_in_train_pairs'] or overlap['test_in_val_pairs']:
        test_keys = key(test_df)
        if args.drop_test_overlap:
            for name, df in [('Train', train_df), ('Val', val_df)]:
                mask = [k in test_keys for k in zip(df['API_CID'], df['Excipient_CID'])]
                log(f"  --drop-test-overlap: removing {sum(mask)} test pairs from {name}")
            train_df = train_df[[k not in test_keys for k in zip(train_df['API_CID'], train_df['Excipient_CID'])]].reset_index(drop=True)
            val_df = val_df[[k not in test_keys for k in zip(val_df['API_CID'], val_df['Excipient_CID'])]].reset_index(drop=True)
        else:
            log("  WARNING: some test pairs are in train/val (kept; pass --drop-test-overlap to remove). "
                "Predictions for those pairs are not unseen-data predictions.")
    run_info['overlap'] = overlap

    if args.smoke:
        train_df = train_df.groupby('Outcome1', group_keys=False).apply(
            lambda g: g.sample(min(len(g), 200 if g.name == 0 else 60), random_state=seed)).reset_index(drop=True)
        val_df = val_df.groupby('Outcome1', group_keys=False).apply(
            lambda g: g.sample(min(len(g), 120 if g.name == 0 else 30), random_state=seed)).reset_index(drop=True)
        class_weights = [main_cw]
        log(f"  SMOKE MODE: train {train_df.shape}, val {val_df.shape}, class weights {class_weights}")

    # ------------------------------------------------------------------ PHASE 1
    log("\n" + "=" * 70)
    log("PHASE 1: Feature engineering (mol2vec + 2D descriptors)")
    log("=" * 70)
    from gensim.models import word2vec
    w2vec_model = word2vec.Word2Vec.load(os.path.join(BASE_DIR, 'model_300dim.pkl'))
    log(f"  Mol2vec embedding: vector_size={w2vec_model.wv.vector_size}")
    with open(os.path.join(BASE_DIR, 'variables.txt'), 'r') as f:
        selected_descriptors = [c for c in f.read().strip().split(',') if c]
    log(f"  2D descriptors from variables.txt: {len(selected_descriptors)}")

    feats = {}
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        x_mol, x_2d, valid = load_or_compute_features(df, name, dirs['features'], w2vec_model, selected_descriptors)
        feats[name] = {'mol2vec': x_mol, '2D': x_2d, 'valid': valid,
                       'y': df.loc[valid, 'Outcome1'].values.astype(int)}
    if len(feats['test']['valid']) != len(test_df):
        raise RuntimeError("Some test pairs could not be featurised - fix their SMILES before running.")

    imputer = SimpleImputer(strategy='median').fit(feats['train']['2D'])
    for name in feats:
        feats[name]['2D'] = imputer.transform(feats[name]['2D'])
    joblib.dump(imputer, os.path.join(dirs['models'], 'imputer_2D.joblib'))
    run_info['timings_s']['features'] = time.time() - t_start

    y_train, y_val, y_test = feats['train']['y'], feats['val']['y'], feats['test']['y']

    # SVMSMOTE per branch on train only (branch models), as in the paper
    resampled = {}
    for branch in ['mol2vec', '2D']:
        Xr, yr = SVMSMOTE(random_state=seed).fit_resample(feats['train'][branch], y_train)
        resampled[branch] = (Xr, yr)
        log(f"  Post-SMOTE {branch}: {Xr.shape}, class dist {np.bincount(yr)}")

    # ------------------------------------------------------------------ PHASE 2
    log("\n" + "=" * 70)
    log("PHASE 2: 8-algorithm grid search (full paper grids) on both branches")
    log("=" * 70)
    t0 = time.time()
    grids = PAPER_GRIDS
    if args.smoke:
        grids = {n: {k: v[:1] for k, v in g.items()} for n, g in PAPER_GRIDS.items()}

    gs_results, gs_tables = {}, []
    for branch in ['mol2vec', '2D']:
        log(f"\n--- {branch} branch ---")
        gs_results[branch] = {}
        Xr, yr = resampled[branch]
        for i, (name, grid) in enumerate(grids.items(), 1):
            log(f"  [{i}/8] {name} ({branch})")
            res = run_grid_search(name, grid, Xr, yr, branch, seed, args.n_jobs, dirs['grid_search'])
            gs_results[branch][name] = res
            tbl = res['cv_results'].copy()
            tbl.insert(0, 'algorithm', name)
            tbl.insert(0, 'branch', branch)
            gs_tables.append(tbl)
    pd.concat(gs_tables, ignore_index=True).to_csv(os.path.join(out_dir, 'grid_search_results.csv'), index=False)
    run_info['timings_s']['grid_search'] = time.time() - t0

    # Collect probabilities of every model on val / test (and train for GHOST on combined)
    probs = {}      # model_name -> {'val': p, 'test': p}
    model_meta = {}  # model_name -> descriptive fields

    for branch in ['mol2vec', '2D']:
        ranking = []
        for name, res in gs_results[branch].items():
            m = res['model']
            mname = f"{branch}/{name}"
            probs[mname] = {s: m.predict_proba(feats[s][branch])[:, 1] for s in ['val', 'test']}
            model_meta[mname] = {'family': 'base', 'branch': branch, 'algorithm': name,
                                 'class_weight': '', 'params': json.dumps(res['best_params'], default=str)}
            ranking.append((name, accuracy_score(y_val, m.predict(feats['val'][branch])), res['best_cv_accuracy']))
        log(f"\n  {branch} ranking by val accuracy (paper's selection criterion):")
        for r, (name, va, cv) in enumerate(sorted(ranking, key=lambda t: -t[1]), 1):
            log(f"    #{r} {name:<20} val acc {va:.4f}  (CV {cv:.4f})")

    # ------------------------------------------------------------------ PHASE 3
    log("\n" + "=" * 70)
    log("PHASE 3: Stacking models (AdaBoost + RF + XGBoost -> LR), class-weight ablation")
    log("=" * 70)
    t0 = time.time()
    stacks = {}
    for cw in class_weights:
        for branch in ['mol2vec', '2D']:
            Xr, yr = resampled[branch]
            stack = train_stack(Xr, yr, gs_results[branch], cw, branch, seed, args.n_jobs, dirs['models'])
            stacks[(branch, cw)] = stack
            mname = f"{branch}/stack_cw{cw[0]:.1f}:{cw[1]:.1f}"
            probs[mname] = {s: stack.predict_proba(feats[s][branch])[:, 1] for s in ['val', 'test']}
            model_meta[mname] = {'family': 'stack', 'branch': branch, 'algorithm': 'Ada+RF+XGB->LR',
                                 'class_weight': f"{cw[0]:.1f}:{cw[1]:.1f}", 'params': ''}
            log(f"      val F1@0.5 = {f1_score(y_val, (probs[mname]['val'] >= 0.5).astype(int)):.4f}")
    run_info['timings_s']['stacks'] = time.time() - t0

    # ------------------------------------------------------------------ PHASE 4
    log("\n" + "=" * 70)
    log("PHASE 4: Combined model (LR on [p_2D, p_mol2vec]) for each class weight")
    log("=" * 70)
    t0 = time.time()
    # Joint SVMSMOTE so both branch inputs of each synthetic row describe the same pair
    n_mol = feats['train']['mol2vec'].shape[1]
    X_joint, y_joint = SVMSMOTE(random_state=seed).fit_resample(
        np.concatenate([feats['train']['mol2vec'], feats['train']['2D']], axis=1), y_train)
    log(f"  Post-SMOTE joint: {X_joint.shape}, class dist {np.bincount(y_joint)}")

    ghost_thresholds = {}
    for cw in class_weights:
        s_mol, s_2d = stacks[('mol2vec', cw)], stacks[('2D', cw)]
        # As in the authors' code: meta-learner fit on branch predictions of the (resampled) TRAIN set
        meta_train = np.column_stack([s_2d.predict_proba(X_joint[:, n_mol:])[:, 1],
                                      s_mol.predict_proba(X_joint[:, :n_mol])[:, 1]])
        lr = LogisticRegression(random_state=seed, max_iter=1000).fit(meta_train, y_joint)
        cw_s = f"{cw[0]:.1f}:{cw[1]:.1f}"
        mname = f"combined/cw{cw_s}"
        joblib.dump(lr, os.path.join(dirs['models'], f"combined_lr_cw{cw[0]:.1f}-{cw[1]:.1f}.joblib"))
        probs[mname] = {}
        for s in ['val', 'test']:
            meta = np.column_stack([s_2d.predict_proba(feats[s]['2D'])[:, 1],
                                    s_mol.predict_proba(feats[s]['mol2vec'])[:, 1]])
            probs[mname][s] = lr.predict_proba(meta)[:, 1]
        model_meta[mname] = {'family': 'combined', 'branch': 'mol2vec+2D', 'algorithm': 'LR(stack_2D, stack_mol2vec)',
                             'class_weight': cw_s, 'params': json.dumps({'coef': lr.coef_.round(4).tolist(),
                                                                         'intercept': lr.intercept_.round(4).tolist()})}
        ghost_thresholds[mname] = {}
        try:
            ghost_thresholds[mname]['ghost_train'] = ghost_threshold(
                y_joint, lr.predict_proba(meta_train)[:, 1], seed, args.ghost_subsets)
            ghost_thresholds[mname]['ghost_val'] = ghost_threshold(
                y_val, probs[mname]['val'], seed, args.ghost_subsets)
        except Exception as e:
            log(f"  WARNING: GHOST failed for {mname}: {e}")
        log(f"  {mname}: coef {lr.coef_.round(3).tolist()}, GHOST {ghost_thresholds[mname]}")
    run_info['timings_s']['combined'] = time.time() - t0

    # ------------------------------------------------------------------ PHASE 5
    log("\n" + "=" * 70)
    log("PHASE 5: Threshold ablations, metrics and 24-pair predictions")
    log("=" * 70)
    main_model = f"combined/cw{main_cw[0]:.1f}:{main_cw[1]:.1f}"
    pair_label = (test_df['Api_name'].astype(str) + ' + ' + test_df['Excipient_name'].astype(str)
                  if {'Api_name', 'Excipient_name'} <= set(test_df.columns)
                  else test_df['API_CID'].astype(str) + ' + ' + test_df['Excipient_CID'].astype(str))

    metric_rows, long_rows, sweep_rows = [], [], []
    sweep = np.round(np.arange(0, 1.0005, 0.001), 3)
    for mname, p in probs.items():
        rules = {
            'default_0.5': 0.5,
            'f1_val': f1_optimal_threshold(y_val, p['val']),
            'paper_f1_0.202': PAPER_F1_THRESHOLD,
            'paper_ghost_0.020': PAPER_GHOST_THRESHOLD,
        }
        rules.update(ghost_thresholds.get(mname, {}))
        for rule, t in rules.items():
            for split, y in [('val', y_val), ('test', y_test)]:
                metric_rows.append({'model': mname, **model_meta[mname], 'threshold_rule': rule,
                                    'split': split, **compute_metrics(y, p[split], t)})
            for i in range(len(y_test)):
                pred = int(p['test'][i] >= t)
                long_rows.append({'model': mname, 'threshold_rule': rule, 'threshold': t,
                                  'pair': pair_label.iloc[i], 'API_CID': test_df['API_CID'].iloc[i],
                                  'Excipient_CID': test_df['Excipient_CID'].iloc[i],
                                  'pair_in_train': in_train[i], 'ground_truth': int(y_test[i]),
                                  'prob': p['test'][i], 'pred': pred, 'correct': int(pred == y_test[i])})
        for split, y in [('val', y_val), ('test', y_test)]:
            for t in sweep:
                pr = p[split] >= t
                tp = int(np.sum(pr & (y == 1))); fp = int(np.sum(pr & (y == 0)))
                fn = int(np.sum(~pr & (y == 1))); tn = int(np.sum(~pr & (y == 0)))
                denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
                sweep_rows.append({'model': mname, 'split': split, 'threshold': t,
                                   'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp,
                                   'accuracy': (tp + tn) / len(y),
                                   'precision': tp / (tp + fp) if tp + fp else 0.0,
                                   'recall': tp / (tp + fn) if tp + fn else 0.0,
                                   'f1_score': 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                                   'mcc': (tp * tn - fp * fn) / denom if denom else 0.0})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(out_dir, 'metrics_all.csv'), index=False)
    summary = metrics_df[metrics_df['split'] == 'test'].drop(columns=['split'])
    summary.to_csv(os.path.join(out_dir, 'summary_24_pairs.csv'), index=False)
    pd.DataFrame(long_rows).to_csv(os.path.join(out_dir, 'predictions_24_long.csv'), index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(out_dir, 'threshold_sweep.csv'), index=False)

    wide = test_df[[c for c in ['Api_name', 'Excipient_name', 'API_CID', 'Excipient_CID'] if c in test_df.columns]].copy()
    wide['ground_truth'] = y_test
    wide['pair_in_train'] = in_train
    for mname, p in probs.items():
        wide[mname] = p['test']
    wide.to_csv(os.path.join(out_dir, 'predictions_24_wide.csv'), index=False)

    # ------------------------------------------------------------------ SUMMARY
    run_info['timings_s']['total'] = time.time() - t_start
    run_info['versions'] = {
        'python': platform.python_version(), 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__,
        'rdkit': rdBase.rdkitVersion, 'numpy': np.__version__, 'pandas': pd.__version__,
    }
    run_info['n_rows'] = {s: len(feats[s]['valid']) for s in feats}
    run_info['main_model'] = main_model
    with open(os.path.join(out_dir, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=2, default=str)

    cols = ['threshold', 'incompatible_found', 'compatible_correct', 'precision', 'recall',
            'f1_score', 'mcc', 'roc_auc', 'pr_auc']
    pd.set_option('display.width', 250)
    log("\n" + "=" * 70)
    log(f"MAIN MODEL ({main_model}) on the 24 pairs, all threshold rules")
    log("=" * 70)
    print(summary[summary['model'] == main_model].set_index('threshold_rule')[cols].round(3).to_string())

    log("\n" + "=" * 70)
    log("ALL MODELS on the 24 pairs, threshold tuned for F1 on val (paper method)")
    log("=" * 70)
    print(summary[summary['threshold_rule'] == 'f1_val'].set_index('model')[cols].round(3).to_string())

    log("\n" + "=" * 70)
    log(f"PER-PAIR probabilities ({main_model}, both branch stacks)")
    log("=" * 70)
    cw_s = f"{main_cw[0]:.1f}:{main_cw[1]:.1f}"
    per_pair = pd.DataFrame({
        'pair': pair_label.values, 'gt': y_test, 'in_train': in_train,
        'p_2D_stack': probs[f'2D/stack_cw{cw_s}']['test'],
        'p_mol2vec_stack': probs[f'mol2vec/stack_cw{cw_s}']['test'],
        'p_combined': probs[main_model]['test'],
    })
    print(per_pair.round(3).to_string(index=False))
    log(f"\nAll outputs written to {out_dir}  (total {run_info['timings_s']['total'] / 60:.1f} min)")


if __name__ == '__main__':
    main()
