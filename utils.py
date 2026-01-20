import inspect
from datasets import Dataset
import os
import errno
import torch
import json
import random
import re
import rank_bm25
import numpy as np
from pathlib import Path
import torch.distributed as dist
from itertools import chain
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from sentence_transformers import SentenceTransformer, util

def set_signature_columns_if_needed(model):
    # Inspect model forward signature to keep only the arguments it accepts.
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += list(set(["labels"]))
    
    return signature_columns

def remove_unused_columns(model, dataset: "Dataset"):

    
    signature_columns = set_signature_columns_if_needed(model)

    ignored_columns = list(set(dataset.column_names) - set(signature_columns))

    columns = [k for k in signature_columns if k in dataset.column_names]

    return dataset.remove_columns(ignored_columns)
    



class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio)*step/float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
            1.0 + (self.min_ratio - 1) * (step - self.warmup_steps)/float(max(1.0, self.scheduler_steps - self.warmup_steps)),
        )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    def lr_lambda(self, step):
        return 1.0


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def set_optim(opt, model):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == 'linear':
        if opt.scheduler_steps is None:
            scheduler_steps = opt.total_steps
        else:
            scheduler_steps = opt.scheduler_steps
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps, scheduler_steps=scheduler_steps, min_ratio=0., fixed_lr=opt.fixed_lr)
    return optimizer, scheduler


def average_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if opt.is_main:
            x = x / opt.world_size
    return x


def sum_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def weighted_average(x, count, opt):
    if not opt.is_distributed:
        return x, count
    t_loss = torch.tensor([x * count], device=opt.device)
    t_total = torch.tensor([count], device=opt.device)
    t_loss = sum_main(t_loss, opt)
    t_total = sum_main(t_total, opt)
    return (t_loss / t_total).item(), t_total.item()


class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}
        
    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result
    

def retrieve_m2c_zero_prompts(molecule):
    template = "Generate a caption for the molecule: {}\n".format(molecule)

    return template


def retrieve_m2c_prompts(examples, molecule, reverse=True):

    def get_template(num):
        template = "Generate a caption for the molecule: {}\nCaption: {}\n\n".format(examples[num]["molecule"], examples[num]["caption"])
        return template
    
    example_prompts = ""
    for i in range(len(examples)):
        if reverse:
            example_prompts = get_template(i) + example_prompts
        else:
            example_prompts += get_template(i)

    full_prompt = example_prompts + "Based on the above examples, analyse the similarities and differences between the examples and finally generate a caption for the molecule: {}. \n".format(molecule)
    
    return full_prompt

def retrieve_bace_prompts(examples, molecule, reverse=True):

    def get_template(num):
        template = "Predict whether the qualitative binding result is active for a set of inhibitors of human β-secretase 1(BACE-1) for the molecule: {}\nPrediction:{}\n\n".format(examples[num]["molecule"], examples[num]["property"])
        return template
    
    example_prompts = ""
    for i in range(len(examples)):
        if reverse:
            example_prompts = get_template(i) + example_prompts
        else:
            example_prompts += get_template(i)

    full_prompt = example_prompts + "Based on the above examples, analyse the similarities and differences between the examples and finally predict whether the qualitative binding result is active for a set of inhibitors of human β-secretase 1(BACE-1) for the molecule: {}\n".format(molecule)
    
    return full_prompt

def retrieve_bbbp_prompts(examples, molecule, reverse=True):
    
    def get_template(num):
        template = "Predict whether the molecule: {} has the property of blood-brain barrier penetration(permeability)\nPrediction:{}\n\n".format(examples[num]["molecule"], examples[num]["property"])
        return template
    
    example_prompts = ""
    for i in range(len(examples)):
        if reverse:
            example_prompts = get_template(i) + example_prompts
        else:
            example_prompts += get_template(i)

    full_prompt = example_prompts + "Based on the above examples, analyse the similarities and differences between the examples and finally predict whether the molecule: {} has the property of blood-brain barrier penetration(permeability)\n".format(molecule)
    
    return full_prompt

def retrieve_tox21_prompts(examples, molecule, target, reverse=True):
        
    def get_template(num):
        template = "Predict whether the molecule: {} can activate or affect {}\nPrediction:{}\n\n".format(examples[num]["molecule"], target, examples[num]["property"])
        return template
    
    example_prompts = ""
    for i in range(len(examples)):
        if reverse:
            example_prompts = get_template(i) + example_prompts
        else:
            example_prompts += get_template(i)

    full_prompt = example_prompts + "Based on the above examples, analyse the similarities and differences between the examples and finally predict whether the molecule: {} can activate or affect {}\n".format(molecule, target)
    
    return full_prompt

def retrieve_sider_prompts(examples, molecule, target, reverse=True):
            
    def get_template(num):
        template = "Predict whether the molecule: {} causes the side effect of {}\nPrediction:{}\n\n".format(examples[num]["molecule"], target, examples[num]["property"])
        return template
    
    example_prompts = ""
    for i in range(len(examples)):
        if reverse:
            example_prompts = get_template(i) + example_prompts
        else:
            example_prompts += get_template(i)

    full_prompt = example_prompts + "Based on the above examples, analyse the similarities and differences between the examples and finally predict whether the molecule: {} causes the side effect of {}\n".format(molecule, target)
    
    return full_prompt

def retrieve_clintox_prompts(examples, molecule, target, reverse=True):

    if target == "fda":
                    
        def get_template(num):
            template = "Predict whether the drug molecule: {} is approved by the FDA\nPrediction:{}\n\n".format(examples[num]["molecule"], examples[num]["fda"])
            return template
        
        example_prompts = ""
        for i in range(len(examples)):
            if reverse:
                example_prompts = get_template(i) + example_prompts
            else:
                example_prompts += get_template(i)

        full_prompt = example_prompts + "Based on the above examples, analyse the similarities and differences between the examples and finally predict whether the drug molecule: {} can be approved by the FDA\n".format(molecule)
    elif target == "ctt":
        def get_template(num):
            template = "Predict whether the drug molecule: {} failed clinical trials for toxicity reason\nPrediction:{}\n\n".format(examples[num]["molecule"], examples[num]["ctt"])
            return template
        
        example_prompts = ""
        for i in range(len(examples)):
            if reverse:
                example_prompts = get_template(i) + example_prompts
            else:
                example_prompts += get_template(i)

        full_prompt = example_prompts + "Based on the above examples, analyse the similarities and differences between the examples and finally predict whether the drug molecule: {} failed clinical trials for toxicity reason\n".format(molecule)
    return full_prompt  

def retrieve_hiv_prompts(examples, molecule, reverse=True):
                                
    def get_template(num):
        template = "Predict whether the molecule: {} inhibits HIV replication\nPrediction:{}\n\n".format(examples[num]["molecule"], examples[num]["property"])
        return template
    
    example_prompts = ""
    for i in range(len(examples)):
        if reverse:
            example_prompts = get_template(i) + example_prompts
        else:
            example_prompts += get_template(i)

    full_prompt = example_prompts + "Based on the above examples, analyse the similarities and differences between the examples and finally predict whether the molecule: {} inhibits HIV replication\n".format(molecule)
    
    return full_prompt

def retrieve_toxcast_prompts(examples, molecule, target, reverse=True):
                                        
    def get_template(num):
        template = "Predict whether the molecule: {} has the property of {}\nPrediction:{}\n\n".format(examples[num]["molecule"], target, examples[num]["property"])
        return template
    
    example_prompts = ""
    for i in range(len(examples)):
        if reverse:
            example_prompts = get_template(i) + example_prompts
        else:
            example_prompts += get_template(i)

    full_prompt = example_prompts + "Based on the above examples, analyse the similarities and differences between the examples and finally predict whether the molecule: {} has the property of {}\n".format(molecule, target)
    
    return full_prompt

def retrieve_c2m_zero_prompts(caption):
    template = "Generate a molecule for the caption: {}\n".format(caption)
    return template


def retrieve_c2m_prompts(examples, caption, reverse=True):

    def get_template(num):
        template = "Generate a molecule for the caption: {}\nMolecule: {}\n\n".format(examples[num]["caption"], examples[num]["molecule"])
        
        return template
    
    example_prompts = ""
    for i in range(len(examples)):
        if reverse:
            example_prompts = get_template(i) + example_prompts
        else:
            example_prompts += get_template(i)
        
    full_prompt = example_prompts + "Based on the above examples, analyse the similarities and differences between the examples and finally generate a molecule for the caption: {}\n".format(caption)

    return full_prompt

def sentenceBERT_similarity(caption, caption_corpus):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    query_embedding = model.encode([caption], convert_to_tensor=True)
    caption_embeddings = model.encode(caption_corpus, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, caption_embeddings)[0]
    cos_scores = cos_scores.cpu().detach().numpy()
    return cos_scores


def get_examples(file, n_shot, input=None, m2c_method="random", c2m_method="random", molecule_rdkits=None):
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
            cap_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

    elif c2m_method == "sentencebert":
        # retrieve caption examples
        doc_scores = sentenceBERT_similarity(input["caption"], caption_corpus)
        candidates = [i for i in range(len(doc_scores))]
        candidates = sorted(candidates, key=lambda i: doc_scores[i], reverse=True)
        candidates = candidates[:n_shot]
        for candidate in candidates:
            cap_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

    elif c2m_method == "random":
        candidates = random.sample(range(len(lines)), n_shot)
        for candidate in candidates:
            cap_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})
    

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
            mol_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

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
            mol_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})
    
    elif m2c_method == "random":
        candidates = random.sample(range(len(lines)), n_shot)
        for candidate in candidates:
            mol_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

    return cap_examples, mol_examples


# def graph_sampling(top_k, n_neighbours):
#     pass

# import networkx as nx
# import random
# import numpy as np

# # 创建一个带权重的图
# G = nx.Graph()

# # 添加节点和边
# for i in range(100):  # 假设我们有100个节点
#     G.add_node(i)
#     neighbours = random.sample(range(100), 10)  # 为每个节点随机选择10个邻居
#     for neighbour in neighbours:
#         G.add_edge(i, neighbour, weight=random.random())  # 添加带有随机权重的边
# print(G)

# def random_walk(G, start_node, num_steps):
#     walk = [start_node]
#     for _ in range(num_steps):
#         current_node = walk[-1]
#         neighbours = list(G.neighbors(current_node))
#         weights = np.array([G[current_node][neighbour]['weight'] for neighbour in neighbours])
#         normalized_weights = weights / weights.sum()  # 归一化权重
#         next_node = np.random.choice(neighbours, p=normalized_weights)
#         walk.append(next_node)
#     return walk

# def random_walk_sampling(G, start_node, num_steps, top_k):
#     samples = set()
#     while len(samples) < top_k:
#         walk = random_walk(G, start_node, num_steps)
#         samples.add(walk[-1]) 
#     return list(samples)

# unique_samples = random_walk_sampling(G, 0, 10, 10)
# print(unique_samples)

def select_by_similarity_score(candidates, examples, task):
    scores = []
    if task == "mol2cap":
        # print(candidates)
        
        for i in range(len(candidates)):
            cap_scores = sentenceBERT_similarity(candidates[i], [example["caption"] for example in examples])[0]
            # append avg score
            scores.append(np.mean(cap_scores))
        # return the index with highest score
    elif task == "cap2mol":
        # print(candidates)
        for i in range(len(candidates)):
            mol = Chem.MolFromSmiles(candidates[i])
            mol_scores = []
            for example in examples:
                # morgan fingerprint
                if candidates[i] == example["molecule"].lstrip("[START_I_SMILES]").rstrip("[END_I_SMILES]"):
                    mol_scores.append(0)
                    continue
                try:
                    mol_scores.append(DataStructs.FingerprintSimilarity(FingerprintMols.FingerprintMol(mol), FingerprintMols.FingerprintMol(Chem.MolFromSmiles(example["molecule"].lstrip("[START_I_SMILES]").rstrip("[END_I_SMILES]")))))
                except:
                    mol_scores.append(0)
            
            # append avg score
            scores.append(np.mean(mol_scores))
        # return the index with highest score
    else:
        raise ValueError("Task not supported!")
    return np.argmax(scores)