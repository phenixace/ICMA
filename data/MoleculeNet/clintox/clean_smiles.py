import re
import pandas as pd
from rdkit import Chem

def clean_smiles(smiles):
    cleaned_smiles = smiles
    # Define allowed characters in a SMILES string
    # allowed_chars = "BCNOPSFIbcnopsf*()[]-=#@$+123456789%."
    
    # # Filter out illegal characters
    # cleaned_smiles = ''.join([char for char in smiles if char in allowed_chars])
    
    # Optionally, you could use regex to remove specific illegal patterns
    # Example: Remove "*" from the cleaned SMILES
    cleaned_smiles = re.sub(r'\*', '', cleaned_smiles)
    
    return cleaned_smiles




def get_temp_dataset(data_name="clintox", split="train"):
    df = pd.read_csv(f"raw/{data_name}_{split}.csv")
    smiles = df['smiles'].tolist()
    rdkit_mol_objs_list = []
    data_list = []
    id_list = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(clean_smiles(s), sanitize=True )
        if mol is not None:
            rdkit_mol_objs_list.append(mol)
            id_list.append(i)
        else:
            print(f"Invalid SMILES: {s} at index {i}")


get_temp_dataset()
