import random

random.seed(42)

dataset = "hiv"
raw_file = "./MoleculeNet/{}/raw/{}.csv".format(dataset, dataset)

with open(raw_file, "r") as f:
    lines = f.readlines()
head = lines[0]
lines = lines[1:] 
random.shuffle(lines)

train_file = "./MoleculeNet/{}/raw/{}_train.csv".format(dataset, dataset)
valid_file = "./MoleculeNet/{}/raw/{}_validation.csv".format(dataset, dataset)
test_file = "./MoleculeNet/{}/raw/{}_test.csv".format(dataset, dataset)

with open(train_file, "w+") as f:
    f.writelines(head)
    f.writelines(lines[:int(0.8 * len(lines))])

with open(valid_file, "w+") as f:
    f.writelines(head)
    f.writelines(lines[int(0.8 * len(lines)):int(0.9 * len(lines))])

with open(test_file, "w+") as f:
    f.writelines(head)
    f.writelines(lines[int(0.9 * len(lines)):])