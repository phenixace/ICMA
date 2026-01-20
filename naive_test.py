from dataset import Mol2CaptionDataset, ZeroShotDataset
from transformers import AutoTokenizer
from evaluations.text_translation_metrics import text_evaluate
from evaluations.mol_translation_metrics import mol_evaluate
from evaluations.fingerprint_metrics import molfinger_evaluate
from sklearn.metrics import roc_auc_score
# from evaluations.fcd_metric import fcd_evaluate
import argparse
import numpy as np


tokenizer = AutoTokenizer.from_pretrained('laituan245/molt5-base-smiles2caption')

parser = argparse.ArgumentParser()
# add raw_folder, pro_folder, dataset_type
parser.add_argument('--raw_folder', type=str, default='./data/ChEBI-20/raw/')
parser.add_argument('--target_folder', type=str, default='predictions/')
parser.add_argument('--model', type=str, default='galactica-125M-T')
parser.add_argument('--ckp', type=int, default=8000)
parser.add_argument('--task', type=str, default='cap2mol')
parser.add_argument('--dataset_type', type=str, default='test')

args = parser.parse_args()

raw_folder = args.raw_folder


if args.task == 'mol2cap':
    pro_file = './{}/{}/output_pred_{}_{}_processed.txt'.format(args.target_folder, args.model, args.task, args.ckp)
    test_set = Mol2CaptionDataset(raw_folder, pro_file, args.dataset_type)

    print('Sanity Check')
    print('test set size:{}'.format(len(test_set)))
    print('test set sample:{}'.format(test_set[0]))

    targets = []
    preds = []
    molecules = []
    for i in range(len(test_set)):
        molecules.append(test_set[i][0])
        targets.append(test_set[i][1])
        preds.append(test_set[i][2])

    metrics = text_evaluate(tokenizer, targets, preds, molecules, 256)

    print('Metrics: bleu-2:{}, bleu-4:{}, rouge-1:{}, rouge-2:{}, rouge-l:{}, meteor-score:{}'.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))


# test_set = CalibratedResult(pro_folder, pro_folder + 'c2m/one_shot/', 'c2m', 'test')
elif args.task == 'cap2mol':
    pro_file = './{}/{}/output_pred_{}_{}_processed.txt'.format(args.target_folder, args.model, args.task, args.ckp)
    test_set = Mol2CaptionDataset(raw_folder, pro_file, args.dataset_type)

    print('Sanity Check')
    print('test set size:{}'.format(len(test_set)))
    print('test set sample:{}'.format(test_set[0]))
    targets = []
    preds = []
    descriptions = []

    for i in range(len(test_set)):
        descriptions.append(test_set[i][1])
        targets.append(test_set[i][0])
        preds.append(test_set[i][3])

    metrics = mol_evaluate(targets, preds, descriptions)
    finger_metrics = molfinger_evaluate(targets, preds)
    # print(targets[0], preds[0])
    # fcd_metrics= fcd_evaluate(targets, preds)
    print("Metrics: bleu_score:{}, em-score:{}, levenshtein:{}, maccs fts:{}, rdk fts:{}, morgan fts:{}, validity_score:{}".format(metrics[0], metrics[1], metrics[2], finger_metrics[1], finger_metrics[2], finger_metrics[3], metrics[3]))
    # print("FCD Similarity:{}".format(fcd_metrics))

elif args.task in ["bace", "bbbp", "clintox", "hiv", "muv", "toxcast", "sider", "tox21"]:
    pro_file = './{}/{}/output_pred_{}_{}.txt'.format(args.target_folder, args.model, args.task, args.ckp)
    # TODO: add more tasks
    test_set = ZeroShotDataset(args.raw_folder, args.task, True, "test") 
    print('Sanity Check')
    print('test set size:{}'.format(len(test_set)))
    print('test set sample:{}'.format(test_set[0]))
    targets = []
    preds = []
    print(len(test_set))
    for i in range(len(test_set)):
        # print(test_set.targets_only[i])
        if test_set.targets_only[i] == "Yes":
            targets.append(1)
        else:
            targets.append(0)
        
    with open(pro_file, "r") as f:
        for line in f:
            tmp = line.strip().strip('\n').strip().split('\t')[1]
            preds.append(float(tmp))
    # classification metric: auc
    targets = np.array(targets)
    preds = np.array(preds)
    # print(targets, preds)
    metric = roc_auc_score(targets, preds)
    print("Metrics: auc:{}".format(metric))
else:
    raise NotImplementedError("Task {} is not implemented".format(args.task))