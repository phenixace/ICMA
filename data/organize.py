'''
Organize the multi-processing data
'''

folder = "PubChem324k/iupac"
task = "cap2mol" # or "mol2cap"
mode = "test"

m2c_method = "random"
c2m_method = "bm25"

for i in range(1, 17):
    with open("./{}/raw/{}_10_shot_mol2cap_{}.txt".format(folder, mode, m2c_method), "a+") as f:
        with open("./{}/raw/{}_10_shot_Part_{}_mol2cap_{}.txt".format(folder, mode, i, m2c_method), "r") as f2:
            lines = f2.readlines()
            for line in lines:
                f.write(line)

    with open("./{}/raw/{}_10_shot_cap2mol_{}.txt".format(folder, mode, c2m_method), "a+") as f:
        with open("./{}/raw/{}_10_shot_Part_{}_cap2mol_{}.txt".format(folder, mode, i, c2m_method), "r") as f2:
            lines = f2.readlines()
            for line in lines:
                f.write(line)