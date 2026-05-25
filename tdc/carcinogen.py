from tdc.single_pred import Tox
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

data = Tox(name = 'Carcinogens_Lagunin')
all_drugs = data.get_data()['Drug']


x_1 = all_drugs[0]
x_2 = all_drugs[1]

#Calculate tanimoto similarity between two molecules
mol1 = Chem.MolFromSmiles(x_1)
mol2 = Chem.MolFromSmiles(x_2)
fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
tanimoto_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
print(f'Tanimoto similarity between {x_1} and {x_2} is {tanimoto_sim}')