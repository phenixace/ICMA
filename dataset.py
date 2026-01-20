from torch.utils.data import Dataset
from rdkit import Chem
from utils import get_examples, retrieve_m2c_prompts, retrieve_c2m_prompts, retrieve_bace_prompts, retrieve_bbbp_prompts, retrieve_clintox_prompts, retrieve_hiv_prompts, retrieve_sider_prompts, retrieve_tox21_prompts, retrieve_toxcast_prompts
import json
import random

class DPODataset(Dataset):
    def __init__(self, data_folder, task, add_eos, mode, add_special_token=False, retrieval=False) -> None:
        '''
        add_special_token: whether to add special token to indicate the SMILES strings
        '''
        super().__init__()
        self.prompt = []
        self.chosen = []
        self.rejected = []
        if task == "mol2cap":
            print("Loading data from {}".format(data_folder + mode + ".txt"))
            with open(data_folder + mode + ".txt", "r", encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[1:]:
                _, mol, cap = line.split("\t")
                if add_special_token:
                    mol = "Generate a caption for the molecule: [START_I_SMILES]{}[END_I_SMILES]\n".format(mol.strip())
                else:
                    mol = "Generate a caption for the molecule: {}\n".format(mol.strip())
                self.prompt.append(mol)
                if add_eos:
                    cap  = mol +"Caption: {}{}".format(cap.strip(), add_eos)
                else:
                    cap  = mol +"Caption: {}".format(cap.strip())
                self.chosen.append(cap)
                # TODO: add rejected
            with open(data_folder + mode + "_rejected.txt", "r", encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[1:]:
                _, _, rej_cap = line.split("\t")
                self.rejected.append(rej_cap)
                
        elif task == "cap2mol":
            print("Loading data from {}".format(data_folder + mode + ".txt"))
            with open(data_folder + mode + ".txt", "r", encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[1:]:
                _, mol, cap = line.split("\t")
                cap = "Generate corresponding molecule based on the caption: {}\n".format(cap.strip())
                self.prompt.append(cap)

                if add_special_token:
                    if add_eos:
                        mol = "Molecule: [START_I_SMILES]{}[END_I_SMILES]{}".format(mol.strip(), add_eos)
                    else:
                        mol = "Molecule: [START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())
                else:
                    if add_eos:
                        mol = cap + "Molecule: {}{}".format(mol.strip(), add_eos)
                    else:
                        mol = cap + "Molecule: {}".format(mol.strip())
                self.chosen.append(mol)
                # TODO: add rejected
            with open(data_folder + mode + "_rejected.txt", "r", encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[1:]:
                _, rej_mol, _ = line.split("\t")

                self.rejected.append(rej_mol)
        elif task == "instruct":
            print("Loading data from {}".format(data_folder + mode + ".txt"))
            with open(data_folder + mode + ".txt", "r", encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[1:]:
                _, mol, cap = line.split("\t")

                if add_special_token:
                    mol_prompt = "Generate a caption for the molecule: [START_I_SMILES]{}[END_I_SMILES]\n".format(mol.strip())
                else:
                    mol_prompt = "Generate a caption for the molecule: {}\n".format(mol.strip())
                cap_prompt = "Generate corresponding molecule based on the caption: {}\n".format(cap.strip())
                self.prompt.append(mol_prompt)
                self.prompt.append(cap_prompt)
                
                if add_eos:
                    cap_tgt  = mol_prompt +"Caption: {}{}".format(cap.strip(), add_eos)
                else:
                    cap_tgt  = mol_prompt +"Caption: {}".format(cap.strip())
                self.chosen.append(cap_tgt)
                
                if add_special_token:
                    if add_eos:
                        mol_tgt = "Molecule: [START_I_SMILES]{}[END_I_SMILES]{}".format(mol.strip(), add_eos)
                    else:
                        mol_tgt = "Molecule: [START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())
                else:
                    if add_eos:
                        mol_tgt = cap_prompt +"Molecule: {}{}".format(mol.strip(), add_eos)
                    else:
                        mol_tgt = cap_prompt +"Molecule: {}".format(mol.strip())
                self.chosen.append(mol_tgt)
                
        else:
            raise NotImplementedError
                

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index], self.rejected[index]

    def __len__(self) -> int:
        return len(self.data)

# Legacy Dataset for testing
class Mol2CaptionDataset(Dataset):
    def __init__(self, raw_folder, pro_file, mode):
        raw_file = raw_folder + '/{}.txt'.format(mode)
        with open(raw_file, 'r') as f:
            lines = f.readlines()

        lines = lines[1:]
        self.data = []
        for line in lines:
            temp = line.strip().split('\t')
            self.data.append([temp[-2], temp[-1]])

        with open(pro_file, 'r') as f:
            lines = f.readlines()
        lines = lines[1:]
        
        for idx in range(len(lines)):
            temp = lines[idx].strip().split('\t')
            try:
                self.data[idx].extend([temp[-2], temp[-1]])
            except:
                print(idx)
                exit(0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # format: [molecule, caption, pred_caption, pred_molecule]
        return self.data[idx]

class ZeroShotDataset(Dataset):
    def __init__(self, data_folder, task, add_eos, mode, add_special_token=False) -> None:
        super().__init__()
        self.data = []
        self.targets = []
        self.targets_only = []
        if task == "mol2cap":
            print("Loading data from {}".format(data_folder + mode + ".txt"))
            with open(data_folder + mode + ".txt", "r", encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[1:]:
                _, mol, cap = line.split("\t")
                if add_special_token:
                    mol_prompt = "## User: Generate a caption for the molecule: [START_I_SMILES]{}[END_I_SMILES]\n".format(mol.strip())
                else:
                    mol_prompt = "Generate a caption for the molecule: {}\n".format(mol.strip())
                self.data.append(mol_prompt)
                self.targets_only.append(cap.strip())

                if add_eos:
                    cap_tgt  = mol_prompt +"Caption: {}{}".format(cap.strip(), add_eos)
                else:
                    cap_tgt  = mol_prompt +"Caption: {}".format(cap.strip())
                self.targets.append(cap_tgt)
        elif task == "cap2mol":
            print("Loading data from {}".format(data_folder + mode + ".txt"))
            with open(data_folder + mode + ".txt", "r", encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[1:]:
                _, mol, cap = line.split("\t")
                mol_o = mol
                if add_special_token:
                    mol = "[START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())
                    cap_prompt = "## User: Generate a molecule that: {}\n## Assistant: ".format(cap.strip())
                else:
                    cap_prompt = "Generate a molecule that: {}\n".format(cap.strip())
                self.data.append(cap_prompt)
                self.targets_only.append(mol_o.strip())
                if add_eos:
                    if add_special_token:
                        mol_tgt = cap_prompt + "{}{}".format(mol.strip(), add_eos)
                    else:
                        mol_tgt = cap_prompt + "Molecule: {}{}".format(mol.strip(), add_eos)
                else:
                    mol_tgt = cap_prompt + "Molecule: {}".format(mol.strip())
                self.targets.append(mol_tgt)
        elif task == "instruct":
            print("Loading data from {}".format(data_folder + mode + ".txt"))
            with open(data_folder + mode + ".txt", "r", encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[1:]:
                _, mol, cap = line.split("\t")
                if add_special_token:
                    mol_prompt = "Generate a caption for the molecule: [START_I_SMILES]{}[END_I_SMILES]\n".format(mol.strip())
                else:
                    mol_prompt = "Generate a caption for the molecule: {}\n".format(mol.strip())
                cap_prompt = "Generate corresponding molecule based on the caption: {}\n".format(cap.strip())
                self.data.append(mol_prompt)
                self.data.append(cap_prompt)

                self.targets_only.append(cap.strip())
                self.targets_only.append(mol.strip())

                if add_eos:
                    cap_tgt  = mol_prompt +"Caption: {}{}".format(cap.strip(), add_eos)
                else:
                    cap_tgt  = mol_prompt +"Caption: {}".format(cap.strip())
                self.targets.append(cap_tgt)
                
                if add_special_token:
                    if add_eos:
                        mol_tgt = "Molecule: [START_I_SMILES]{}[END_I_SMILES]{}".format(mol.strip(), add_eos)
                    else:
                        mol_tgt = "Molecule: [START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())
                else:
                    if add_eos:
                        mol_tgt = cap_prompt +"Molecule: {}{}".format(mol.strip(), add_eos)
                    else:
                        mol_tgt = cap_prompt +"Molecule: {}".format(mol.strip())
                self.targets.append(mol_tgt)
        elif task == "bbbp":
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            with open(raw_file, 'r') as f:
                lines = f.readlines()
            for line in lines[1:]:
                mol = line.strip().split(',')[-1]
                prop = line.strip().split(',')[-2]
                if prop == "":
                    continue
                if add_special_token:
                    mol_prompt = "Predict whether the molecule: [START_I_SMILES]{}[END_I_SMILES] has the property of blood-brain barrier penetration(permeability). (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip())
                else:
                    mol_prompt = "Predict whether the molecule: {} has the property of blood-brain barrier penetration(permeability). (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip())
                self.data.append(mol_prompt)
                self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                if add_eos:
                    prop_tgt  = mol_prompt +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                else:
                    prop_tgt  = mol_prompt +"{}".format("Yes" if prop.strip()=="1" else "No")
                self.targets.append(prop_tgt)
        elif task == "bace":
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            with open(raw_file, 'r') as f:
                lines = f.readlines()
            for line in lines[1:]:
                mol = line.strip().split(',')[0]
                prop = line.strip().split(',')[2]
                if prop == "":
                    continue
                if add_special_token:
                    mol_prompt = "Predict whether the qualitative binding result is active for a set of inhibitors of human β-secretase 1(BACE-1) for the molecule: [START_I_SMILES]{}[END_I_SMILES]. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip())
                else:
                    mol_prompt = "Predict whether the qualitative binding result is active for a set of inhibitors of human β-secretase 1(BACE-1) for the molecule: {}. (Yout answer should be like: Prediction:Yes or Prediction:No)\nPrediction:".format(mol.strip())
                self.data.append(mol_prompt)
                self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                if add_eos:
                    prop_tgt  = mol_prompt +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                else:
                    prop_tgt  = mol_prompt +"{}".format("Yes" if prop.strip()=="1" else "No")
                self.targets.append(prop_tgt)
        elif task == "clintox":
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            with open(raw_file, 'r') as f:
                lines = f.readlines()
            for line in lines[1:]:
                mol, fda, ctt = line.strip().split(',')
                if fda.strip() != "":
                    if add_special_token:
                        mol_prompt_fda = "Predict whether the drug molecule: [START_I_SMILES]{}[END_I_SMILES] is approved by the FDA. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip())
                    else:
                        mol_prompt_fda = "Predict whether the drug molecule: {} is approved by the FDA. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip())
                    
                    self.data.append(mol_prompt_fda)

                    self.targets_only.append("Yes" if fda.strip()=="1" else "No")
                    if add_eos:
                        fda_tgt  = mol_prompt_fda +"{}{}".format("Yes" if fda.strip()=="1" else "No", add_eos)
                    else:
                        fda_tgt  = mol_prompt_fda +"{}".format("Yes" if fda.strip()=="1" else "No")

                    self.targets.append(fda_tgt)
                if ctt.strip() != "":
                    if add_special_token:
                        mol_prompt_ctt = "Predict whether the drug molecule: [START_I_SMILES]{}[END_I_SMILES] failed clinical trials for toxicity reason. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip())
                    else:
                        mol_prompt_ctt = "Predict whether the drug molecule: {} failed clinical trials for toxicity reason. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip())
                    self.data.append(mol_prompt_ctt)
                    self.targets_only.append("Yes" if ctt.strip()=="1" else "No")
                    if add_eos:
                        ctt_tgt  = mol_prompt_ctt +"{}{}".format("Yes" if ctt.strip()=="1" else "No", add_eos)
                    else:
                        ctt_tgt  = mol_prompt_ctt +"{}".format("Yes" if ctt.strip()=="1" else "No")

                    self.targets.append(ctt_tgt)
        elif task == "hiv":
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            with open(raw_file, 'r') as f:
                lines = f.readlines()
            for line in lines[1:]:
                mol = line.strip().split(',')[0]
                prop = line.strip().split(',')[2]
                if prop == "":
                    continue
                if add_special_token:
                    mol_prompt = "Predict whether the molecule: [START_I_SMILES]{}[END_I_SMILES] inhibits HIV replication. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip())
                else:
                    mol_prompt = "Predict whether the molecule: {} inhibits HIV replication. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip())
                self.data.append(mol_prompt)
                self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                if add_eos:
                    prop_tgt  = mol_prompt +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                else:
                    prop_tgt  = mol_prompt +"{}".format("Yes" if prop.strip()=="1" else "No")
                self.targets.append(prop_tgt)
        # elif task == "muv":
        #     pass
        elif task == "toxcast":
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            with open(raw_file, 'r') as f:
                lines = f.readlines()
            bio_tgts = lines[0].split(',')[1:]

            for line in lines[1:]:
                mol = line.strip().split(',')[0]
                properties = line.strip().split(',')[1:]
                for prop, bio_tgt in zip(properties, bio_tgts):
                    if prop == "":
                        continue
                    if add_special_token:
                        mol_prompt = "Predict whether the molecule: [START_I_SMILES]{}[END_I_SMILES] has the property of {}. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip(), bio_tgt.strip())
                    else:
                        mol_prompt = "Predict whether the molecule: {} has the property of {}. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip(), bio_tgt.strip())
                    self.data.append(mol_prompt)
                    self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                    if add_eos:
                        prop_tgt  = mol_prompt +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                    else:
                        prop_tgt  = mol_prompt +"{}".format("Yes" if prop.strip()=="1" else "No")
                    self.targets.append(prop_tgt)

        elif task == "sider":  # done
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            with open(raw_file, 'r') as f:
                lines = f.readlines()
            bio_tgts = lines[0].split(',')[1:]

            for line in lines[1:]:
                mol = line.strip().split(',')[0]
                properties = line.strip().split(',')[1:]
                for prop, bio_tgt in zip(properties, bio_tgts):
                    if prop.strip() == "":
                        continue
                    if add_special_token:
                        mol_prompt = "Predict whether the molecule: [START_I_SMILES]{}[END_I_SMILES] causes the side effect of {}. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip(), bio_tgt.strip())
                    else:
                        mol_prompt = "Predict whether the molecule: {} causes causes the side effect of {}. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip(), bio_tgt.strip())
                    self.data.append(mol_prompt)
                    self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                    if add_eos:
                        prop_tgt  = mol_prompt +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                    else:
                        prop_tgt  = mol_prompt +"{}".format("Yes" if prop.strip()=="1" else "No")
                    self.targets.append(prop_tgt)

        elif task == "tox21":  # done
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            with open(raw_file, 'r') as f:
                lines = f.readlines()
            bio_tgts = ["Androgen receptor using the MDA cell line (NR-AR)", "Androgen receptor ligand binding domain (NR-AR-LBD)", "Aryl hydrocarbon receptor (NR-AhR)", 
                        "Aromatase enzyme (NR-Aromatase)", "Estrogen receptor alpha using the BG1 cell line (NR-ER)", "Estrogen receptor alpha ligand binding domain (NR-ER-LBD)",
                        "Peroxisome proliferator-activated receptor gamma (NR-PPAR-gamma)", "Antioxidant response element (SR-ARE)", "Luciferase-tagged ATAD5 in human embryonic kidney cells (SR-ATAD5)",
                        "Heat shock response (SR-HSE)", "Mitochondrial membrane potential (SR-MMP)", "p53 transcription factor response (SR-p53)"]
            for line in lines[1:]:
                mol = line.strip().split(',')[-1]
                properties = line.strip().split(',')[:-2]
                for prop, bio_tgt in zip(properties, bio_tgts):
                    if prop.strip() == "":
                        continue
                    if add_special_token:
                        mol_prompt = "Predict whether the molecule: [START_I_SMILES]{}[END_I_SMILES] can activate or affect {}. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip(), bio_tgt.strip())
                    else:
                        mol_prompt = "Predict whether the molecule: {} can activate or affect {}. (Yout answer should be like: Prediction:Yes or Prediction:No).\nPrediction:".format(mol.strip(), bio_tgt.strip())
                    self.data.append(mol_prompt)
                    self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                    if add_eos:
                        prop_tgt  = mol_prompt +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                    else:
                        prop_tgt  = mol_prompt +"{}".format("Yes" if prop.strip()=="1" else "No")
                    self.targets.append(prop_tgt)

        else:
            raise NotImplementedError
                

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)
    
class RetrievalDataset(Dataset):
    def __init__(self, data_folder, task, add_eos, mode, n_shot=1, add_special_token=False, m2c_method="gnn", c2m_method="bm25", start_pos=0, bucket_sampling=False, N=9) -> None:    # N is actually 10, but we consider the id from 0, so write 9
        # reverse is now default True (removed as parameter)
        reverse = True
        super().__init__()
        self.data = []
        self.targets = []
        self.targets_only = []
        self.examples = []
        if task == "mol2cap":
            print("Loading data from {}".format(data_folder + mode + ".txt"))
            with open(data_folder + mode + ".txt", "r", encoding='utf-8') as f:
                lines = f.readlines()

            db = dict()
            with open(data_folder + "train.txt", "r", encoding='utf-8') as f:
                db_lines = f.readlines()
                for line in db_lines:
                    cid, mol, cap = line.split("\t")
                    db[cid] = {"molecule": mol.strip(), "caption": cap.strip()}

            with open(data_folder + mode + "_10_shot_{}_{}.txt".format(task, m2c_method), "r", encoding='utf-8') as f:
                neighbour_lines = f.readlines()

            
            for i in range(1, len(lines)):
                cid, mol, cap = lines[i].split("\t")
                if add_special_token:
                    mol = "[START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())

                mol_ex = []
                j = start_pos
                num_ex = 0
                while num_ex < n_shot and j < 10:
                    if not bucket_sampling:
                        if j >= 10:
                            break
                        temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]
                        if add_special_token:
                            mol_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "caption": db[temp_id]["caption"]})
                        else:
                            mol_ex.append({"molecule": db[temp_id]["molecule"], "caption": db[temp_id]["caption"]})
                        j = j + 1
                        num_ex = num_ex + 1
                    else:
                        if (j-start_pos) % int((10-start_pos)/n_shot) == 0:
                            temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]
                            if add_special_token:
                                mol_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "caption": db[temp_id]["caption"]})
                            else:
                                mol_ex.append({"molecule": db[temp_id]["molecule"], "caption": db[temp_id]["caption"]})
                            j = j + 1
                            num_ex = num_ex + 1
                        else:
                            j = j + 1


                self.examples.append(mol_ex)
                mol = retrieve_m2c_prompts(mol_ex, mol.strip(), reverse)
                self.data.append(mol.strip())

                self.targets_only.append(cap.strip())
                if add_eos:
                    cap  = mol +"Caption: {}{}".format(cap.strip(), add_eos)
                else:
                    cap  = mol +"Caption: {}".format(cap.strip())
                self.targets.append(cap.strip())
        elif task == "cap2mol":
            print("Loading data from {}".format(data_folder + mode + ".txt"))
            with open(data_folder + mode + ".txt", "r", encoding='utf-8') as f:
                lines = f.readlines()

            db = dict()
            with open(data_folder + "train.txt", "r", encoding='utf-8') as f:
                db_lines = f.readlines()
                for line in db_lines:
                    cid, mol, cap = line.split("\t")
                    db[cid] = {"molecule": mol.strip(), "caption": cap.strip()}

            with open(data_folder + mode + "_10_shot_{}_{}.txt".format(task, c2m_method), "r", encoding='utf-8') as f:
                neighbour_lines = f.readlines()

            
            for i in range(1, len(lines)):
                cid, mol, cap = lines[i].split("\t")
                mol_o = mol
                if add_special_token:
                    mol = "[START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())

                cap_ex = []
                j = start_pos
                num_ex = 0
                while num_ex < n_shot:
                    if not bucket_sampling:
                        if j >= 10:
                            break
                        temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]
                        if add_special_token:
                            cap_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "caption": db[temp_id]["caption"]})
                        else:
                            cap_ex.append({"molecule": db[temp_id]["molecule"], "caption": db[temp_id]["caption"]})
                        j = j + 1
                        num_ex = num_ex + 1
                    else:
                        if (j-start_pos) % int((10-start_pos)/n_shot) == 0:
                            temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]
                            if add_special_token:
                                cap_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "caption": db[temp_id]["caption"]})
                            else:
                                cap_ex.append({"molecule": db[temp_id]["molecule"], "caption": db[temp_id]["caption"]})
                            j = j + 1
                            num_ex = num_ex + 1
                        else:
                            j = j + 1
                self.examples.append(cap_ex)
                cap = retrieve_c2m_prompts(cap_ex, cap.strip(), reverse)
                self.data.append(cap.strip())
                self.targets_only.append(mol_o.strip())
                if add_eos:
                    mol = cap +"Molecule: {}{}".format(mol.strip(), add_eos)
                else:
                    mol = cap +"Molecule: {}".format(mol.strip())
                self.targets.append(mol.strip())
        elif task == "instruct":
            print("Loading data from {}".format(data_folder + mode + ".txt"))
            with open(data_folder + mode + ".txt", "r", encoding='utf-8') as f:
                lines = f.readlines()

            db = dict()
            with open(data_folder + "train.txt", "r", encoding='utf-8') as f:
                db_lines = f.readlines()
                for line in db_lines:
                    cid, mol, cap = line.split("\t")
                    db[cid] = {"molecule": mol.strip(), "caption": cap.strip()}

            with open(data_folder + mode + "_10_shot_{}_{}.txt".format("mol2cap", m2c_method), "r", encoding='utf-8') as f:
                mol2cap_neighbour_lines = f.readlines()

            with open(data_folder + mode + "_10_shot_{}_{}.txt".format("cap2mol", c2m_method), "r", encoding='utf-8') as f:
                cap2mol_neighbour_lines = f.readlines()

            
            for i in range(1, len(lines)):
                cid, mol, cap = lines[i].split("\t")
                if add_special_token:
                    mol = "[START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())

                mol_ex = []
                cap_ex = []
                j = start_pos
                num_ex = 0
                while num_ex < n_shot:
                    if j >= 10:
                        break
                    temp_id = mol2cap_neighbour_lines[(i-1)*10+j].split("\t")[1]
                    if add_special_token:
                        mol_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "caption": db[temp_id]["caption"]})
                    else:
                        mol_ex.append({"molecule": db[temp_id]["molecule"], "caption": db[temp_id]["caption"]})

                    temp_id = cap2mol_neighbour_lines[(i-1)*10+j].split("\t")[1]
                    if add_special_token:
                        cap_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "caption": db[temp_id]["caption"]})
                    else:
                        cap_ex.append({"molecule": db[temp_id]["molecule"], "caption": db[temp_id]["caption"]})
                    
                    j = j + 1
                    num_ex = num_ex + 1
                    
                mol_prompt = retrieve_m2c_prompts(mol_ex, mol.strip(), reverse)
                cap_prompt = retrieve_c2m_prompts(cap_ex, cap.strip(), reverse)

                self.data.append(mol_prompt)
                self.data.append(cap_prompt)

                self.targets_only.append(cap.strip())
                self.targets_only.append(mol.strip())
                if add_eos:
                    cap_tgt  = mol_prompt +"Caption: {}{}".format(cap.strip(), add_eos)
                else:
                    cap_tgt  = mol_prompt +"Caption: {}".format(cap.strip())
                self.targets.append(cap_tgt)

                if add_eos:
                    mol_tgt = cap_prompt +"Molecule: {}{}".format(mol.strip(), add_eos)
                else:
                    mol_tgt = cap_prompt +"Molecule: {}".format(mol.strip())

                self.targets.append(mol_tgt)
        # retrieval version dataset
        elif task == "bbbp":
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            print("Loading data from {}".format(raw_file))
            with open(raw_file, 'r') as f:
                lines = f.readlines()

            db = dict()
            with open(data_folder + "/{}/raw/{}_train.csv".format(task, task), "r", encoding='utf-8') as f:
                db_lines = f.readlines()
                for i in range(1, len(db_lines)):
                    mol = db_lines[i].split(",")[-1]
                    prop = db_lines[i].split(",")[-2]
                    db[str(i)] = {"molecule": mol.strip(), "property": prop.strip()}

            with open(data_folder + "/{}/raw/{}_10_shot_{}_{}.txt".format(task, mode, task, m2c_method), "r", encoding='utf-8') as f:
                neighbour_lines = f.readlines()

            for i in range(1, len(lines)):
                mol = lines[i].strip().split(',')[-1]
                if add_special_token:
                    mol = "[START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())
                prop = lines[i].strip().split(',')[-2]
                if prop == "":
                    continue

                mol_ex = []
                j = start_pos
                num_ex = 0
                while num_ex < n_shot:
                    temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]
                    if db[temp_id]["property"] == "" and j < 10:   # if the property is empty, skip
                        j = j + 1
                        continue
                    elif j >= 10:
                        break
                    
                    if add_special_token:
                        mol_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "property": "Yes" if db[temp_id]["property"]=="1" else "No"})
                        
                    else:
                        mol_ex.append({"molecule": db[temp_id]["molecule"], "property": "Yes" if db[temp_id]["property"]=="1" else "No"})
                    j = j + 1
                    num_ex = num_ex + 1
                mol = retrieve_bbbp_prompts(mol_ex, mol.strip(), reverse)
                mol += "Prediction:"
                self.data.append(mol.strip())

                self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                if add_eos:
                    prop_tgt  = mol +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                else:
                    prop_tgt  = mol +"{}".format("Yes" if prop.strip()=="1" else "No")
                self.targets.append(prop_tgt)

        elif task == "bace":
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            print("Loading data from {}".format(raw_file))
            with open(raw_file, 'r') as f:
                lines = f.readlines()

            db = dict()
            with open(data_folder + "/{}/raw/{}_train.csv".format(task, task), "r", encoding='utf-8') as f:
                db_lines = f.readlines()
                for i in range(1, len(db_lines)):
                    mol = db_lines[i].split(",")[0]
                    prop = db_lines[i].split(",")[2]
                    db[str(i)] = {"molecule": mol.strip(), "property": prop.strip()}

            with open(data_folder + "/{}/raw/{}_10_shot_{}_{}.txt".format(task, mode, task, m2c_method), "r", encoding='utf-8') as f:
                neighbour_lines = f.readlines()
            
            for i in range(1, len(lines)):
                mol = lines[i].strip().split(',')[0]
                if add_special_token:
                    mol = "[START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())
                prop = lines[i].strip().split(',')[2]
                if prop == "":
                    continue

                mol_ex = []
                j = start_pos
                num_ex = 0
                while num_ex < n_shot:
                    temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]
                    if db[temp_id]["property"] == "" and j < 10:   # if the property is empty, skip
                        j = j + 1
                        continue
                    elif j >= 10:
                        break
                    
                    
                    if add_special_token:
                        mol_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "property": "Yes" if db[temp_id]["property"]=="1" else "No"})
                    else:
                        mol_ex.append({"molecule": db[temp_id]["molecule"], "property": "Yes" if db[temp_id]["property"]=="1" else "No"})
                    j = j + 1
                    num_ex = num_ex + 1
                mol = retrieve_bace_prompts(mol_ex, mol.strip(), reverse)
                mol += "Prediction:"
                self.data.append(mol.strip())

                self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                if add_eos:
                    prop_tgt  = mol +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                else:
                    prop_tgt  = mol +"{}".format("Yes" if prop.strip()=="1" else "No")
                self.targets.append(prop_tgt)

        elif task == "clintox":
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            print("Loading data from {}".format(raw_file))

            db = dict()
            with open(data_folder + "/{}/raw/{}_train.csv".format(task, task), "r", encoding='utf-8') as f:
                db_lines = f.readlines()
                for i in range(1, len(db_lines)):
                    mol, fda, ctt = db_lines[i].split(",")
                    db[str(i)] = {"molecule": mol.strip(), "fda": fda.strip(), "ctt": ctt.strip()}
            print(db.keys())
            with open(data_folder + "/{}/raw/{}_10_shot_{}_{}.txt".format(task, mode, task, m2c_method), "r", encoding='utf-8') as f:
                neighbour_lines = f.readlines()

            with open(raw_file, 'r') as f:
                lines = f.readlines()
            
            for i in range(1, len(lines)):
                mol, fda, ctt = lines[i].strip().split(',')
                if add_special_token:
                    mol = "[START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())
                if fda.strip() != "":
                    mol_ex = []
                    j = start_pos
                    num_ex = 0
                    while num_ex < n_shot:
                        if j >= 10:
                            break
                        temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]
                        if add_special_token:
                            mol_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "fda": "Yes" if db[temp_id]["fda"].strip()=="1" else "No"})
                        else:
                            mol_ex.append({"molecule": db[temp_id]["molecule"], "fda": "Yes" if db[temp_id]["fda"].strip()=="1" else "No"})
                        j = j + 1
                        num_ex = num_ex + 1
                    mol_prompt = retrieve_clintox_prompts(mol_ex, mol.strip(), "fda", reverse)

                    mol_prompt += "Prediction:"
                    self.data.append(mol_prompt.strip())

                    self.targets_only.append("Yes" if fda.strip()=="1" else "No")
                    if add_eos:
                        fda_tgt  = mol_prompt +"{}{}".format("Yes" if fda.strip()=="1" else "No", add_eos)
                    else:
                        fda_tgt  = mol_prompt +"{}".format("Yes" if fda.strip()=="1" else "No")
                    self.targets.append(fda_tgt)
                if ctt.strip() != "":
                    mol_ex = []
                    j = start_pos
                    num_ex = 0
                    while num_ex < n_shot:
                        if j >= 10:
                            break
                        temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]
                        if add_special_token:
                            mol_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "ctt": "Yes" if db[temp_id]["ctt"].strip()=="1" else "No"})
                        else:
                            mol_ex.append({"molecule": db[temp_id]["molecule"], "ctt": "Yes" if db[temp_id]["ctt"].strip()=="1" else "No"})
                        j = j + 1
                        num_ex = num_ex + 1
                    mol_prompt = retrieve_clintox_prompts(mol_ex, mol.strip(), "ctt", reverse)
                    mol_prompt += "Prediction:"
                    self.data.append(mol_prompt.strip())

                    self.targets_only.append("Yes" if ctt.strip()=="1" else "No")
                    if add_eos:
                        ctt_tgt  = mol_prompt +"{}{}".format("Yes" if ctt.strip()=="1" else "No", add_eos)
                    else:
                        ctt_tgt  = mol_prompt +"{}".format("Yes" if ctt.strip()=="1" else "No")
                    self.targets.append(ctt_tgt)

        elif task == "hiv":
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            print("Loading data from {}".format(raw_file))
            with open(raw_file, 'r') as f:
                lines = f.readlines()

            db = dict()
            with open(data_folder + "/{}/raw/{}_train.csv".format(task, task), "r", encoding='utf-8') as f:
                db_lines = f.readlines()
                for i in range(1, len(db_lines)):
                    mol = db_lines[i].split(",")[0]
                    prop = db_lines[i].split(",")[2]
                    db[str(i)] = {"molecule": mol.strip(), "property": prop.strip()}

            with open(data_folder + "/{}/raw/{}_10_shot_{}_{}.txt".format(task, mode, task, m2c_method), "r", encoding='utf-8') as f:
                neighbour_lines = f.readlines()

            for i in range(1, len(lines)):
                mol = lines[i].strip().split(',')[0]
                if add_special_token:
                    mol = "[START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())
                prop = lines[i].strip().split(',')[2]
                if prop == "":
                    continue

                mol_ex = []
                j = start_pos
                num_ex = 0
                while num_ex < n_shot:
                
                    temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]
                    if db[temp_id]["property"] == "" and j < 10:   # if the property is empty, skip
                        j = j + 1
                        continue
                    elif j >= 10:
                        break
                    
                    
                    if add_special_token:
                        mol_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "property": "Yes" if db[temp_id]["property"]=="1" else "No"})
                        
                    else:
                        mol_ex.append({"molecule": db[temp_id]["molecule"], "property": "Yes" if db[temp_id]["property"]=="1" else "No"})
                    j = j + 1
                    num_ex = num_ex + 1
                mol = retrieve_hiv_prompts(mol_ex, mol.strip(), reverse)
                mol += "Prediction:"
                self.data.append(mol.strip())

                self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                if add_eos:
                    prop_tgt  = mol +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                else:
                    prop_tgt  = mol +"{}".format("Yes" if prop.strip()=="1" else "No")
                self.targets.append(prop_tgt)

        # elif task == "muv":
        #     pass
        elif task == "toxcast":
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            with open(raw_file, 'r') as f:
                lines = f.readlines()
            bio_tgts = lines[0].split(',')[1:]

            db = dict()
            with open(data_folder + "/{}/raw/{}_train.csv".format(task, task), "r", encoding='utf-8') as f:
                db_lines = f.readlines()
                for i in range(1, len(db_lines)):
                    mol = db_lines[i].split(",")[0]
                    prop = db_lines[i].split(",")[1:]
                    db[str(i)] = {"molecule": mol.strip(), "property": prop}

            with open(data_folder + "/{}/raw/{}_10_shot_{}_{}.txt".format(task, mode, task, m2c_method), "r", encoding='utf-8') as f:
                neighbour_lines = f.readlines()


            for i in range(1, len(lines)):
                mol = lines[i].strip().split(',')[0]
                if add_special_token:
                    mol = "[START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())
                properties = lines[i].strip().split(',')[1:]
                prop_id = 0
                for prop, bio_tgt in zip(properties, bio_tgts):
                    if prop.strip() == "":
                        prop_id = prop_id + 1
                        continue

                    mol_ex = []
                    j = start_pos
                    num_ex = 0
                    while num_ex < n_shot:
                        temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]

                        if db[temp_id]["property"] == "" and j < 10:   # if the property is empty, skip
                            j = j + 1
                            continue
                        elif j >= 10:
                            break
                        
                    
                        if add_special_token:
                            mol_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "property": "Yes" if db[temp_id]["property"][prop_id]=="1" else "No", "target": bio_tgt})
                            
                        else:
                            mol_ex.append({"molecule": db[temp_id]["molecule"], "property": "Yes" if db[temp_id]["property"][prop_id]=="1" else "No", "target": bio_tgt})
                        j = j + 1
                        num_ex = num_ex + 1
                    mol_prompt = retrieve_toxcast_prompts(mol_ex, mol.strip(), bio_tgt, reverse)
                    mol_prompt += "Prediction:"
                    self.data.append(mol_prompt.strip())

                    self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                    if add_eos:
                        prop_tgt  = mol_prompt +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                    else:
                        prop_tgt  = mol_prompt +"{}".format("Yes" if prop.strip()=="1" else "No")
                    self.targets.append(prop_tgt)

        elif task == "sider":  # done
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            print("Loading data from {}".format(raw_file))
            with open(raw_file, 'r') as f:
                lines = f.readlines()
            bio_tgts = lines[0].split(',')[1:]

            db = dict()
            with open(data_folder + "/{}/raw/{}_train.csv".format(task, task), "r", encoding='utf-8') as f:
                db_lines = f.readlines()
                for i in range(1,len(db_lines)):
                    mol = db_lines[i].split(",")[0]
                    prop = db_lines[i].split(",")[1:]
                    db[str(i)] = {"molecule": mol.strip(), "property": prop}

            with open(data_folder + "/{}/raw/{}_10_shot_{}_{}.txt".format(task, mode, task, m2c_method), "r", encoding='utf-8') as f:
                neighbour_lines = f.readlines()

            for i in range(1, len(lines)):
                mol = lines[i].strip().split(',')[0]
                if add_special_token:
                    mol = "[START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())
                properties = lines[i].strip().split(',')[1:]
                prop_id = 0
                for prop, bio_tgt in zip(properties, bio_tgts):
                    if prop.strip() == "":
                        prop_id = prop_id + 1
                        continue

                    mol_ex = []
                    j = start_pos
                    num_ex = 0
                    while num_ex < n_shot:
                        temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]

                        if db[temp_id]["property"] == "" and j < 10:   # if the property is empty, skip
                            j = j + 1
                            continue
                        elif j >= 10:
                            break
                        
                        
                        if add_special_token:
                            mol_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "property": "Yes" if db[temp_id]["property"][prop_id]=="1" else "No", "target": bio_tgt})
                            
                        else:
                            mol_ex.append({"molecule": db[temp_id]["molecule"], "property": "Yes" if db[temp_id]["property"][prop_id]=="1" else "No", "target": bio_tgt})
                        j = j + 1
                        num_ex = num_ex + 1
                    mol_prompt = retrieve_sider_prompts(mol_ex, mol.strip(), bio_tgt, reverse)
                    mol_prompt += "Prediction:"
                    self.data.append(mol_prompt.strip())

                    self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                    if add_eos:
                        prop_tgt  = mol_prompt +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                    else:
                        prop_tgt  = mol_prompt +"{}".format("Yes" if prop.strip()=="1" else "No")
                    self.targets.append(prop_tgt)

        elif task == "tox21":  # done
            raw_file = data_folder + '/{}/raw/{}_{}.csv'.format(task, task, mode)
            with open(raw_file, 'r') as f:
                lines = f.readlines()
            bio_tgts = ["Androgen receptor using the MDA cell line (NR-AR)", "Androgen receptor ligand binding domain (NR-AR-LBD)", "Aryl hydrocarbon receptor (NR-AhR)", 
                        "Aromatase enzyme (NR-Aromatase)", "Estrogen receptor alpha using the BG1 cell line (NR-ER)", "Estrogen receptor alpha ligand binding domain (NR-ER-LBD)",
                        "Peroxisome proliferator-activated receptor gamma (NR-PPAR-gamma)", "Antioxidant response element (SR-ARE)", "Luciferase-tagged ATAD5 in human embryonic kidney cells (SR-ATAD5)",
                        "Heat shock response (SR-HSE)", "Mitochondrial membrane potential (SR-MMP)", "p53 transcription factor response (SR-p53)"]
            
            db = dict()
            with open(data_folder + "/{}/raw/{}_train.csv".format(task, task), "r", encoding='utf-8') as f:
                db_lines = f.readlines()
                for i in range(1, len(db_lines)):
                    mol = db_lines[i].split(",")[-1]
                    prop = db_lines[i].split(",")[:-2]
                    db[str(i)] = {"molecule": mol.strip(), "property": prop}

            with open(data_folder + "/{}/raw/{}_10_shot_{}_{}.txt".format(task, mode, task, m2c_method), "r", encoding='utf-8') as f:
                neighbour_lines = f.readlines()
            
            for i in range(1, len(lines)):
                mol = lines[i].strip().split(',')[-1]
                if add_special_token:
                    mol = "[START_I_SMILES]{}[END_I_SMILES]".format(mol.strip())
                properties = lines[i].strip().split(',')[:-2]

                prop_id = 0
                for prop, bio_tgt in zip(properties, bio_tgts):
                    if prop.strip() == "":
                        prop_id = prop_id + 1
                        continue

                    mol_ex = []
                    j = start_pos
                    num_ex = 0
                    while num_ex < n_shot:
                        temp_id = neighbour_lines[(i-1)*10+j].split("\t")[1]

                        if db[temp_id]["property"] == "" and j < 10:   # if the property is empty, skip
                            j = j + 1
                            continue
                        elif j >= 10:
                            break
                        
                        
                        if add_special_token:
                            mol_ex.append({"molecule": "[START_I_SMILES]{}[END_I_SMILES]".format(db[temp_id]["molecule"]), "property": "Yes" if db[temp_id]["property"][prop_id]=="1" else "No", "target": bio_tgt})
                            
                        else:
                            mol_ex.append({"molecule": db[temp_id]["molecule"], "property": "Yes" if db[temp_id]["property"][prop_id]=="1" else "No", "target": bio_tgt})
                        j = j + 1
                        num_ex = num_ex + 1
                    mol_prompt = retrieve_tox21_prompts(mol_ex, mol.strip(), bio_tgt, reverse)
                    mol_prompt += "Prediction:"
                    self.data.append(mol_prompt.strip())

                    self.targets_only.append("Yes" if prop.strip()=="1" else "No")
                    if add_eos:
                        prop_tgt  = mol_prompt +"{}{}".format("Yes" if prop.strip()=="1" else "No", add_eos)
                    else:
                        prop_tgt  = mol_prompt +"{}".format("Yes" if prop.strip()=="1" else "No")
                    self.targets.append(prop_tgt)

        else:
            raise NotImplementedError

                

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)
    

class PretrainDataset(Dataset):
    def __init__(self, data_folder) -> None:
        super().__init__()

    def __getitem__(self, index: int):
        pass

    def __len__(self) -> int:
        pass

class LLMCollator(object):
    def __init__(self, tokenizer, cutoff_len, padding="longest"):
        self.tokenizer = tokenizer
        self.cut_off_len = cutoff_len
        self.padding = padding

    def __call__(self, batch):
        # batch : [(input, label)]
        # inputs = [ex[0] for ex in batch]
        labels = [ex[1] for ex in batch]

        # tokenize
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            labels,
            max_length=self.cut_off_len,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_labels = self.tokenizer.batch_encode_plus(
            labels,
            max_length=self.cut_off_len,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_labels["input_ids"],
        }

if __name__ == "__main__":
    # dt = ZeroShotDataset("./data/MoleculeNet/", "toxcast", False, "test")
    # dt = RetrievalDataset("./data/MoleculeNet/", "sider", False, "test", n_shot=2, add_special_token=False, m2c_method="gnn", c2m_method="bm25", random_walk=True, start_pos=1)

    dt = RetrievalDataset("./data/ChEBI-20/raw/", "cap2mol", add_eos=True, mode="train", n_shot=4, add_special_token=False, m2c_method="gnn", c2m_method="bm25", reverse=True, random_walk=True, start_pos=1)
    print(len(dt))
    print(dt[0])
    
    # for i in range(len(dt)):
    #     res = tokenizer.encode(dt[i][1])
    #     if len(res) > 2048:
    #         print(dt[i][1])
    #         print(len(res))
    #         print(i)
    #         continue

    # from transformers import AutoConfig
    # config = AutoConfig.from_pretrained("facebook/galactica-125M")
    # print(config)
    # from transformers import AutoTokenizer
    # tk = AutoTokenizer.from_pretrained("facebook/galactica-125M")
    # tk.save_pretrained("./temp/")