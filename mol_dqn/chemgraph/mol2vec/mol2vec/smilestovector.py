from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from gensim.models import word2vec
import timeit
from joblib import Parallel, delayed


from mol2vec.mol2vec.features import *
from mol2vec.mol2vec.helpers import *

from gensim.models import word2vec

#with open("cleaned_data.txt", "r") as f:
#    data = f.readlines()
   
#data.pop(0)

#smiles = [line.split(",")[0] for line in data]

#print(smiles[0])

def smilesToVec(smiles, model):
    
    sentence_list = []
    sentence_list.append(mol2alt_sentence(Chem.MolFromSmiles(smiles), 1))

    # model = word2vec.Word2Vec.load('mol2vec/mol2vec/models/model_300dim.pkl')

    vector = sentences2vec(sentence_list, model)
    
    return vector
#    with open("vectors.txt", "w") as f:
#        for vec in vector_list:
#            output = ""
#            for num in vec:
#                output = output + str(num) + " "
#            output = output + "\n"
#            f.writelines(output)
