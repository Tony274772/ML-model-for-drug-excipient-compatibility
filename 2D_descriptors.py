#%%
import pandas as pd
import numpy as np
from tqdm import tqdm

# Fix #1: correct filename (was 'data.csv', actual file is 'dataset.csv')
data = pd.read_csv('dataset.csv')

# Extract SMILES columns
molecules_API = data["API_Smiles"]
molecules_exp = data["Excipient_Smiles"]

#%%
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def getMolDescriptors(mol, missingVal=None):
    """Compute all RDKit 2D descriptors for a molecule."""
    res = {}
    for nm, fn in Descriptors._descList:
        # some of the descriptor functions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

#%%
# Fix #18: Added error handling around SMILES parsing so one bad molecule
# doesn't crash the whole loop. Also added tqdm progress bar.
API = []
for i, smiles in enumerate(tqdm(molecules_API, desc="Computing API descriptors")):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  WARNING: Failed to parse API SMILES at row {i}: {smiles}")
            API.append(getMolDescriptors(None, missingVal=None))
            continue
        des = getMolDescriptors(mol)
        API.append(des)
    except Exception as e:
        print(f"  ERROR at API row {i}: {e}")
        API.append(getMolDescriptors(None, missingVal=None))

#%%
exp = []
for i, smiles in enumerate(tqdm(molecules_exp, desc="Computing Excipient descriptors")):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  WARNING: Failed to parse Excipient SMILES at row {i}: {smiles}")
            exp.append(getMolDescriptors(None, missingVal=None))
            continue
        des = getMolDescriptors(mol)
        exp.append(des)
    except Exception as e:
        print(f"  ERROR at Excipient row {i}: {e}")
        exp.append(getMolDescriptors(None, missingVal=None))

#%%
# Create DataFrames for the 2D descriptors
df_API = pd.DataFrame(API)
df_exp = pd.DataFrame(exp)
data_2D = pd.concat([df_API, df_exp, data], axis=1)
print(f"2D data shape: {data_2D.shape}")

#%%
# Save the concatenated data to a CSV file
data_2D.to_csv('2D_data.csv', index=False)
print("Saved 2D_data.csv")
