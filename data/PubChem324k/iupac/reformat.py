# reformat the dataset to the format of cap2mol and mol2cap

mode = "validation"
raw_file = "./raw/{}.txt".format(mode)

new_file = "./raw/{}_reformat.txt".format(mode)

with open(raw_file, "r") as f:
    lines = f.readlines()

with open(new_file, "w+") as f:
    f.write("cid\tSMILES\tdescription\n")
    for line in lines[1:]:
        temp = line.strip().strip("\n").strip().split("\t")

        cid = temp[1].lstrip("[Compound(").rstrip(")]")
        smiles = temp[0].strip().strip("\n").strip()
        caption = temp[2].strip().strip("\n").strip()
        if caption == "None":
            continue
        f.write("{}\t{}\t{}\n".format(cid, smiles, caption))