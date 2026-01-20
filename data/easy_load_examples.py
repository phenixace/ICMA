'''
Prepare examples for train set

Could be merged in the future
'''

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from sentence_transformers import SentenceTransformer, util
import rank_bm25
import random
import re
import multiprocessing as mp
import random

def sentenceBERT_similarity(caption, caption_embeddings, model):
    query_embedding = model.encode([caption], convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, caption_embeddings)[0]
    cos_scores = cos_scores.cpu().detach().numpy()
    return cos_scores

def get_examples(file, n_shot, input=None, m2c_method="random", c2m_method="random", molecule_rdkits=None, model=None, caption_embeddings=None):
    with open(file, 'r') as f:
        lines = f.readlines()
    
    lines = lines[1:]
    molecule_corpus = []
    caption_corpus = []
    for line in lines:
        line = line.strip().strip("\n").strip()
        molecule_corpus.append(line.split("\t")[1])
        caption_corpus.append(line.split("\t")[2])

    def remove_punctuation(text):
        text = text.replace("-", " ")
        text = text.replace(",", " ")
        text = text.replace(".", "")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = re.sub(r' +', ' ', text)
        return text


    cap_examples = []
    if c2m_method == "bm25":
        # retrieve caption examples
        tokenized_caption_corpus = []
        for doc in caption_corpus:
            doc = remove_punctuation(doc)
            tokenized_caption_corpus.append(doc.split(" "))

        bm25 = rank_bm25.BM25Okapi(tokenized_caption_corpus)
        query = input["caption"]
        query = remove_punctuation(query)
        tokenized_query = query.split(" ")
        # print(tokenized_query)

        doc_scores = bm25.get_scores(tokenized_query)
        candidates = [i for i in range(len(doc_scores))]
        candidates = sorted(candidates, key=lambda i: doc_scores[i], reverse=True)
        candidates = candidates[:n_shot]
        for candidate in candidates:
            cap_examples.append({"idx": candidate, "score": doc_scores[candidate], "molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

    elif c2m_method == "sentencebert":
        # retrieve caption examples
        doc_scores = sentenceBERT_similarity(input["caption"], caption_embeddings, model)
        candidates = [i for i in range(len(doc_scores))]
        candidates = sorted(candidates, key=lambda i: doc_scores[i], reverse=True)
        candidates = candidates[:n_shot]
        for candidate in candidates:
            cap_examples.append({"idx": candidate, "score": doc_scores[candidate], "molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

    elif c2m_method == "random":
        candidates = random.sample(range(len(lines)), n_shot)
        for candidate in candidates:
            cap_examples.append({"idx": candidate, "score": 1.0, "molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})
    else:
        raise NotImplementedError
    

    mol_examples = []
    if m2c_method == "bm25":
        # retrieve molecule examples
        tokenized_molecule_corpus = [list(doc) for doc in molecule_corpus]
        bm25 = rank_bm25.BM25Okapi(tokenized_molecule_corpus)
        query = input["molecule"]
        tokenized_query = list(query)
        # print(tokenized_query)

        doc_scores = bm25.get_scores(tokenized_query)
        candidates = [i for i in range(len(doc_scores))]
        candidates = sorted(candidates, key=lambda i: doc_scores[i], reverse=True)
        candidates = candidates[:n_shot]

        for candidate in candidates:
            mol_examples.append({"idx": candidate, "score": doc_scores[candidate], "molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

    elif m2c_method == "morgan":
        # retrieve molecule examples
        input_mol = Chem.MolFromSmiles(input["molecule"])
        mol_scores = []
        for mol in molecule_rdkits:
            mol_scores.append(DataStructs.FingerprintSimilarity(FingerprintMols.FingerprintMol(mol), FingerprintMols.FingerprintMol(input_mol)))

        candidates = [i for i in range(len(mol_scores))]
        candidates = sorted(candidates, key=lambda i: mol_scores[i], reverse=True)
        candidates = candidates[:n_shot]

        for candidate in candidates:
            mol_examples.append({"idx": candidate, "score": mol_scores[candidate], "molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})
    
    elif m2c_method == "random":
        candidates = random.sample(range(len(lines)), n_shot)
        for candidate in candidates:
            mol_examples.append({"idx": candidate, "score": 1.0, "molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})
    else:
        raise NotImplementedError

    return cap_examples, mol_examples

folder = "PubChem324k/iupac"
mode = "test"
raw = "./{}/raw/{}.txt".format(folder, mode)

train = "./{}/raw/{}.txt".format(folder, "train")

n_shot = 10
m2c_method = "random"
c2m_method = "bm25"
model = None
caption_embeddings = None
if c2m_method == "sentencebert":
    with open(train, 'r') as f:
        lines = f.readlines()
    
    lines = lines[1:]
    caption_corpus = []
    for line in lines:
        line = line.strip().strip("\n").strip()
        caption_corpus.append(line.split("\t")[2])

    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    caption_embeddings = model.encode(caption_corpus, convert_to_tensor=True)

random.seed(42)

with open(raw, "r") as f:
    tgt_lines = f.readlines()

with open(train, "r") as f:
    lines = f.readlines()

rdkit_molecules = []
for line in lines[1:]:
    line = line.strip().strip("\n").strip()
    rdkit_molecules.append(Chem.MolFromSmiles(line.split("\t")[1]))

process = 1

def run(n):
    print("Running Part {}".format(n))
    if n == 1:
        for line in tgt_lines[1:int(len(tgt_lines)/process)]:
            temp = line.strip().strip("\n").strip().split("\t")

            cid = temp[0]
            smiles = temp[1]
            caption = temp[2]
            # print(temp)
            cap_ex, mol_ex = get_examples(train, n_shot+1, input={"molecule": smiles, "caption": caption}, m2c_method=m2c_method, c2m_method=c2m_method, molecule_rdkits=rdkit_molecules, model=model, caption_embeddings=caption_embeddings)
            # print(cap_ex)
            # print(cid, cap_ex, mol_ex)
            with open("./{}/raw/{}_{}_shot_Part_{}_mol2cap_{}.txt".format(folder, mode, n_shot, n, m2c_method), "a+") as f:
               
                for i in range(n_shot):
                    mol_ex_idx = lines[mol_ex[i]["idx"]+1].strip().strip("\n").strip().split("\t")[0]
                    f.write("{}\t{}\t{}\n".format(cid, mol_ex_idx, mol_ex[i]["score"]))

            with open("./{}/raw/{}_{}_shot_Part_{}_cap2mol_{}.txt".format(folder, mode, n_shot, n, c2m_method), "a+") as f:
                # print(cap_ex)
                for i in range(1,n_shot+1):
                    cap_ex_idx = lines[cap_ex[i]["idx"]+1].strip().strip("\n").strip().split("\t")[0]
                    f.write("{}\t{}\t{}\n".format(cid, cap_ex_idx, cap_ex[i]["score"]))
            print("Log: Part {}: {}, {}".format(n, mol_ex_idx, cap_ex_idx))
    else:
        for line in tgt_lines[int(len(tgt_lines)*(n-1)/process):int(len(tgt_lines)*n/process)]:
            temp = line.strip().strip("\n").strip().split("\t")

            cid = temp[0]
            smiles = temp[1]
            caption = temp[2]

            cap_ex, mol_ex = get_examples(train, n_shot, input={"molecule": smiles, "caption": caption}, m2c_method=m2c_method, c2m_method=c2m_method, molecule_rdkits=rdkit_molecules, model=model, caption_embeddings=caption_embeddings)
            
            with open("./{}/raw/{}_{}_shot_Part_{}_mol2cap_{}.txt".format(folder, mode, n_shot, n, m2c_method), "a+") as f:
               
                for i in range(n_shot):
                    mol_ex_idx = lines[mol_ex[i]["idx"]+1].strip().strip("\n").strip().split("\t")[0]
                    f.write("{}\t{}\t{}\n".format(cid, mol_ex_idx, mol_ex[i]["score"]))

            with open("./{}/raw/{}_{}_shot_Part_{}_cap2mol_{}.txt".format(folder, mode, n_shot, n, c2m_method), "a+") as f:
               
                for i in range(n_shot):
                    cap_ex_idx = lines[cap_ex[i]["idx"]+1].strip().strip("\n").strip().split("\t")[0]
                    f.write("{}\t{}\t{}\n".format(cid, cap_ex_idx, cap_ex[i]["score"]))
            print("Log: Part {}: {}, {}".format(n, mol_ex_idx, cap_ex_idx))
# initial multiprocessing
# pool = mp.Pool(processes=process)
# for i in range(1, process+1):
#     pool.apply_async(run, args=(i, ))
# pool.close()
# pool.join()
            
run(1)