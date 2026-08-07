#%%
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm

#%%
# Fix #1: correct filename (was 'data.csv', actual file is 'dataset.csv')
df = pd.read_csv('dataset.csv')
print(df.head())
print(f"Dataset shape: {df.shape}")

#%%
# Parse molecules with error handling (Fix #18)
print("Parsing SMILES and adding hydrogens...")
failed_rows = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing SMILES"):
    try:
        mol_api = Chem.MolFromSmiles(row['API_Smiles'])
        mol_exp = Chem.MolFromSmiles(row['Excipient_Smiles'])
        if mol_api is None:
            print(f"  WARNING: Failed to parse API SMILES at row {idx}: {row['API_Smiles']}")
            failed_rows.append(idx)
            continue
        if mol_exp is None:
            print(f"  WARNING: Failed to parse Excipient SMILES at row {idx}: {row['Excipient_Smiles']}")
            failed_rows.append(idx)
            continue
        df.at[idx, 'mol_API'] = mol_api
        df.at[idx, 'mol_Excipient'] = mol_exp
    except Exception as e:
        print(f"  ERROR at row {idx}: {e}")
        failed_rows.append(idx)

if failed_rows:
    print(f"Dropping {len(failed_rows)} rows with failed SMILES parsing")
    df = df.drop(failed_rows).reset_index(drop=True)

# Re-parse after cleanup (object columns don't survive the loop well with .at)
df['mol_API'] = df['API_Smiles'].apply(lambda x: Chem.MolFromSmiles(x))
df['mol_Excipient'] = df['Excipient_Smiles'].apply(lambda x: Chem.MolFromSmiles(x))
df['mol_Excipient'] = df['mol_Excipient'].apply(lambda x: Chem.AddHs(x))
df['mol_API'] = df['mol_API'].apply(lambda x: Chem.AddHs(x))

#%%
# Fix #2: Removed hardcoded Windows path (was os.chdir(r'C:\Users\PC\OneDrive - ...'))
# Fix #3: Removed dead open('model_300dim.pkl', 'rb') call that was never used

from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

# Load mol2vec model from current directory (Fix #15: it's actually 100-dim, not 300)
w2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')
print(f"Mol2vec model loaded: vector_size={w2vec_model.wv.vector_size}")

df['sentence_API'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_API'], 1)), axis=1)
df['mol2vec_API'] = [DfVec(x) for x in sentences2vec(df['sentence_API'], w2vec_model, unseen='UNK')]
df['sentence_Excipient'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_Excipient'], 1)), axis=1)
df['mol2vec_Excipient'] = [DfVec(x) for x in sentences2vec(df['sentence_Excipient'], w2vec_model, unseen='UNK')]

#%%
# Fix #20: Log UNK-fragment rate
vocab = set(w2vec_model.wv.key_to_index.keys())
for col_name, sent_col in [('API', 'sentence_API'), ('Excipient', 'sentence_Excipient')]:
    total_tokens = 0
    unk_tokens = 0
    for sent in df[sent_col]:
        for token in sent:
            total_tokens += 1
            if str(token) not in vocab:
                unk_tokens += 1
    if total_tokens > 0:
        print(f"{col_name} UNK rate: {unk_tokens}/{total_tokens} = {unk_tokens/total_tokens:.4f}")

#%%
X1 = np.array([x.vec for x in df['mol2vec_API']])
X2 = np.array([y.vec for y in df['mol2vec_Excipient']])
X = pd.concat((pd.DataFrame(X1), pd.DataFrame(X2), df.drop(['mol2vec_API', 'mol2vec_Excipient',
                                                              'sentence_Excipient',
                                                              'mol_API', 'mol_Excipient',
                                                              'sentence_API'], axis=1)), axis=1)
print(X)
print(f"Mol2vec data shape: {X.shape}")

#%%
X.to_csv('mol2vec_data.csv', sep=',', encoding='utf-8', index=False)
print("Saved mol2vec_data.csv")
#%%
