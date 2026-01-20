# Calculate the similarity between the graph representations of the train and test data
import torch
import torch.nn.functional as F

folder = 'MoleculeNet/'
task = 'clintox'
if task != 'mol2cap' and task != 'iupac':
    folder = folder + task +"/"
n_shot = 10

if task != 'mol2cap' and task != 'iupac':
    raw_train = "./{}/raw/{}_train.csv".format(folder, task)
    raw_test = "./{}/raw/{}_test.csv".format(folder, task)
else:
    raw_train = "./{}/raw/train.txt".format(folder)
    raw_test = "./{}/raw/test.txt".format(folder)


train_tensors = torch.load('./{}/GNN_SIM/graph_representations_Mole-BERT_{}_train.pt'.format(folder, task))
print(train_tensors.size())

test_tensors = torch.load('./{}/GNN_SIM/graph_representations_Mole-BERT_{}_test.pt'.format(folder, task))
print(test_tensors.size())

with open(raw_train) as f:
    train_data = f.readlines()

with open(raw_test) as f:
    test_data = f.readlines()


if task == 'mol2cap' or task == 'iupac':
    # process train data
    for i in range(len(train_tensors)):
        cos_sim = F.cosine_similarity(train_tensors, train_tensors[i].unsqueeze(0), dim=1)
        # Get the top 10 indices
        topk_values, topk_indices = torch.topk(cos_sim, n_shot+1)
        
        print("Top 10 values: ", topk_values)
        print("Top 10 indices: ", topk_indices)

        cid = train_data[i+1].split('\t')[0]
        topk_indices = topk_indices.squeeze(0).tolist()

        with open('./{}/raw/train_{}_shot_{}_gnn.txt'.format(folder, n_shot, task), 'a+') as f:
            for idx, score in zip(topk_indices, topk_values):
                if idx == i:
                    continue
                f.write(cid + '\t' + train_data[idx+1].split('\t')[0] + '\t' + str(score.item()) + '\n')

    # process test data
    for i in range(len(test_tensors)):
        cos_sim = F.cosine_similarity(train_tensors, test_tensors[i].unsqueeze(0), dim=1)
        # Get the top 10 indices
        topk_values, topk_indices = torch.topk(cos_sim, n_shot)
        
        print("Top 10 values: ", topk_values)
        print("Top 10 indices: ", topk_indices)

        cid = test_data[i+1].split('\t')[0]
        topk_indices = topk_indices.squeeze(0).tolist()
        with open('./{}/raw/test_{}_shot_{}_gnn.txt'.format(folder, n_shot, task), 'a+') as f:
            for idx, score in zip(topk_indices, topk_values):
                f.write(cid + '\t' + train_data[idx+1].split('\t')[0] + '\t' + str(score.item()) + '\n')

else:
    for i in range(len(train_tensors)):
        cos_sim = F.cosine_similarity(train_tensors, train_tensors[i].unsqueeze(0), dim=1)
        # Get the top 10 indices
        topk_values, topk_indices = torch.topk(cos_sim, n_shot+1)
        
        print("Top 10 values: ", topk_values)
        print("Top 10 indices: ", topk_indices)

        cid = str(i+1)
        topk_indices = topk_indices.squeeze(0).tolist()

        with open('./{}/raw/train_{}_shot_{}_gnn.txt'.format(folder, n_shot, task), 'a+') as f:
            for idx, score in zip(topk_indices, topk_values):
                if idx == i:
                    continue
                f.write(cid + '\t' + str(idx+1) + '\t' + str(score.item()) + '\n')

    # process test data
    for i in range(len(test_tensors)):
        cos_sim = F.cosine_similarity(train_tensors, test_tensors[i].unsqueeze(0), dim=1)
        # Get the top 10 indices
        topk_values, topk_indices = torch.topk(cos_sim, n_shot)
        
        print("Top 10 values: ", topk_values)
        print("Top 10 indices: ", topk_indices)

        cid = str(i+1)
        topk_indices = topk_indices.squeeze(0).tolist()
        with open('./{}/raw/test_{}_shot_{}_gnn.txt'.format(folder, n_shot, task), 'a+') as f:
            for idx, score in zip(topk_indices, topk_values):
                f.write(cid + '\t' + str(idx+1) + '\t' + str(score.item()) + '\n')
