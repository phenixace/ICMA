from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
from dataset import Mol2CaptionDataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


with open("./data/ChEBI-20/raw/test.txt") as f:
    test_lines = f.readlines()
cids = []
molecules = []
captions = []
for line in test_lines[1:]:
    cids.append(line.split("\t")[0].strip().strip("\n").strip())
    molecules.append(line.split("\t")[1].strip().strip("\n").strip())
    captions.append(line.split("\t")[2].strip().strip("\n").strip())

print(len(captions))



generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.75,
                top_p=0.85,
                top_k=40,
                num_beams=1,
                pad_token_id=0,
            )
model = T5ForConditionalGeneration.from_pretrained("laituan245/molt5-large-caption2smiles").cuda()
tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-caption2smiles")

with open("./predictions/molt5/output_pred_cap2mol_8000_processed.txt", "w+") as f:
    f.write("cid\tCaption\tSMILES\n")

for cid, cap, mol in zip(cids, captions, molecules):

    inputs = tokenizer(cap, return_tensors="pt", padding=True, truncation=True)["input_ids"].cuda()
    outputs = model.generate(inputs, generation_config=generation_config, max_new_tokens=256)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    with open("./predictions/molt5/output_pred_cap2mol_8000_processed.txt", "a+") as f:
        f.write(cid + "\tN/A\t" + decoded + "\n")