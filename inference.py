import os
# GPU environment
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
import torch
import argparse
import random
import transformers
from datasets import Dataset
from dataset import ZeroShotDataset, LLMCollator, RetrievalDataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from utils import remove_unused_columns, select_by_similarity_score
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, T5ForConditionalGeneration, DataCollatorForSeq2Seq

from peft import (
    PeftModel
)

from rdkit import Chem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="facebook/galactica-1.3b") # facebook/galactica-125m
    parser.add_argument("--adapter_path", type=str, default="./ckp/galactica-1b/mol2cap/checkpoint-12000/")
    parser.add_argument("--data_folder", type=str, default="./data/ChEBI-20/raw/")
    
    parser.add_argument("--task", type=str, default="mol2cap")
    
    parser.add_argument("--model_type", type=str, default="decoder-only")

    parser.add_argument("--output_dir", type=str, default="./predictions/galactica-1b-Retrieval/")
    parser.add_argument("--cutoff_len", type=int, default=1024)  # anyway, reserve 256 tokens for generation
    
    parser.add_argument("--batch_size", type=int, default=1)

    # generation config
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--n_shot", type=int, default=1)
    
    parser.add_argument("--seed", type=int, default=42)
    
    # something new in generation!
    parser.add_argument("--post_review", default=False, action="store_true")
    parser.add_argument("--post_retrieve", default=False, action="store_true")
 
    # retrieval settings
    parser.add_argument("--retrieval", default=False, action="store_true")
    parser.add_argument("--m2c_method", type=str, default="gnn")
    parser.add_argument("--c2m_method", type=str, default="bm25")
    parser.add_argument("--bucket_sampling", default=False, action="store_true")

    parser.add_argument("--disable_lora", default=False, action="store_true")
    parser.add_argument("--batch_infer", default=False, action="store_true")
    parser.add_argument("--add_eos", type=str, default="</s>")
    parser.add_argument("--int8", default=False, action="store_true")
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--add_special_token", default=False, action="store_true")
    parser.add_argument("--start_pos", type=int, default=0)

    args = parser.parse_args()
    
    if not args.post_review:
        args.num_return_sequences = 1    # must be 1 for generation
        
    # check out put dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    if os.path.exists(args.output_dir + "/output_pred_" + args.task + "_" + str(args.adapter_path.split("-")[-1].strip("/")) + ".txt"):
        print("Output file already exists!")
        with open(args.output_dir + "/output_pred_" + args.task + "_" + str(args.adapter_path.split("-")[-1].strip("/")) + ".txt", "r") as f:
            temp = f.readlines()
        start_pos = len(temp)
    else:
        start_pos = 0
    print("Start from: ", start_pos)

        
    # set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    transformers.set_seed(args.seed)
    random.seed(args.seed)

    # print parameters
    print("========Parameters========")
    for attr, value in args.__dict__.items():
        print("{}={}".format(attr.upper(), value))


    # load dataset
    test_dataset = RetrievalDataset(args.data_folder, args.task, args.add_eos, "test", args.n_shot, args.add_special_token, args.m2c_method, args.c2m_method, args.start_pos, args.bucket_sampling) if args.retrieval else ZeroShotDataset(args.data_folder, args.task, args.add_eos, "test", args.add_special_token) 
    test_data = Dataset.from_dict({"gt": test_dataset.targets, "raw": test_dataset.data})
    print(test_data[0])
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # alter tokenizer
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    # tokenizer.truncation_side ='left'
    # tokenizer.add_tokens(train_data.get_special_tokens)
    
    def tokenize(prompt):
        
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
            # no_repeat_ngram_size=10,
        )

        result["labels"] = result["input_ids"].copy()

        return result
        
    def generate_and_tokenize_prompt(data_point):
        # For test
        tokenized_full_prompt = tokenize(data_point['raw'])
        tokenized_label_prompt = tokenize(data_point['gt'])
        tokenized_full_prompt["labels"] = tokenized_label_prompt["input_ids"]
            
        return tokenized_full_prompt
    
    
    test_data = (
        test_data.map(lambda sample: generate_and_tokenize_prompt(sample))
    )

    print("========Sanity Check========")
    print(test_data[1])
    print(tokenizer.decode(test_data[0]["input_ids"]))
    
    
    # load model
    device_map = "auto"
    if args.model_type == "decoder-only":
        if not args.disable_lora:
            model = AutoModelForCausalLM.from_pretrained(args.base_model, load_in_8bit=True if args.int8 else False, torch_dtype=torch.float16 if args.fp16 else torch.float32, device_map=device_map)
            model = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype=torch.float16 if args.fp16 else torch.float32, device_map=device_map)    
        else:
            model = AutoModelForCausalLM.from_pretrained(args.adapter_path, load_in_8bit=True if args.int8 else False, torch_dtype=torch.float16 if args.fp16 else torch.float32, device_map=device_map)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.base_model, load_in_8bit=True if args.int8 else False, torch_dtype=torch.float16 if args.fp16 else torch.float32).cuda()
    # model = model.merge_and_unload()
    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    model.half()
    model.eval() 
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    # load collator
    # collator = LLMRec_Collator(tokenizer, args.cutoff_len, padding="max_length")

    # load dataloader
    # test_dataloader = torch.utils.data.DataLoader(
    #    test_data,
    #    batch_size=1,
    #    shuffle=False,
    #    num_workers=0,
    #    collate_fn=collator,
    #    pin_memory=True,
    # )

    generation_config = GenerationConfig(
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                repetition_penalty=0.8 if args.task=="cap2mol" else 1.0,
                pad_token_id=0,
            )
    
    logits_to_save = []
    # batch inference
    if args.batch_infer:
        test_data = remove_unused_columns(model, test_data)
        collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
        # distributed sampler
        # sampler = DistributedSampler(test_data)
        data_loader = DataLoader(test_data, collate_fn=collator, batch_size=args.batch_size)
        # batch inference
        init_num = 0
        with tqdm(total=len(data_loader)) as pbar:
            for _, batch in enumerate(data_loader):
                batch["input_ids"] = batch["input_ids"].cuda()
                batch["attention_mask"] = batch["attention_mask"].cuda()
                with torch.no_grad():
                    generation = model.generate(
                        **batch,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=args.max_new_tokens,
                        num_return_sequences=args.num_return_sequences
                    )
                    
                for current_num in range(args.batch_size):
                    if args.post_review and args.task == "cap2mol":
                        flag = False
                        candidates = []
                        outputs = []
                        cur_examples = [ex["molecule"] for ex in test_dataset.examples[init_num + current_num]]
                        for idx in range(args.num_return_sequences):
                            s = tokenizer.decode(generation.sequences[current_num*args.num_return_sequences + idx], skip_special_tokens=True)
                            try:
                                tmp = s.split('Molecule')[-1].strip().strip(":").strip('\n').strip()
                            except:
                                print(tmp)
                                continue
                            mol = Chem.MolFromSmiles(tmp)
                            if mol is not None:
                                flag = True
                                candidates.append(tmp)
                                outputs.append(s)
                                
                                if not args.post_retrieve and tmp.strip() not in cur_examples:
                                    break
                        if not flag:    # could not find valid molecule
                            s = tokenizer.decode(generation.sequences[current_num*args.num_return_sequences], skip_special_tokens=True)

                        if args.post_retrieve and flag:
                            # for every candidate in candidates, calculate average the similarity score and retrieve the best one
                            best_idx = select_by_similarity_score(candidates, test_dataset.examples[init_num + current_num], args.task)
                            s = outputs[best_idx]
                    
                    elif args.task == "mol2cap" and args.post_retrieve:
                        flag = False
                        candidates = []
                        outputs = []
                        for idx in range(args.num_return_sequences):
                            s = tokenizer.decode(generation.sequences[current_num*args.num_return_sequences + idx], skip_special_tokens=True)
                            try:
                                tmp = s.split('Caption')[-1].strip(":").strip('\n').strip()
                                flag = True
                            except:
                                print(tmp)
                                continue
                            candidates.append(tmp)
                            outputs.append(s)
                        if not flag:
                            s = tokenizer.decode(generation.sequences[current_num*args.num_return_sequences], skip_special_tokens=True)
                        # for every candidate in candidates, calculate average the similarity score and retrieve the best one
                        else:
                            best_idx = select_by_similarity_score(candidates, test_dataset.examples[init_num + current_num], args.task)
                            s = outputs[best_idx]
                    elif args.task in ["bbbp", "bace", "clintox", "hiv", "muv", "toxcast", "sider", "tox21"]:
                        yes_token_id = tokenizer.convert_tokens_to_ids("Yes") # mistral
                        no_token_id = tokenizer.convert_tokens_to_ids("No")
                        for idx in range(args.num_return_sequences):
                            s = tokenizer.decode(generation.sequences[current_num*args.num_return_sequences + idx], skip_special_tokens=True)
                        
                            scores = generation.scores[current_num*args.num_return_sequences + idx].softmax(dim=-1)
                            logits_to_save.append(scores)
                            logits = torch.tensor(scores[:,[yes_token_id, no_token_id]], dtype=torch.float32).softmax(dim=-1)[0]
                            s += "\t" + str(logits[0].item())
                    else:
                        s = tokenizer.decode(generation.sequences[current_num*args.num_return_sequences], skip_special_tokens=True)
                    print(s)
                    with open(args.output_dir + "/output_pred_" + args.task + "_" + args.adapter_path.split("-")[-1].strip("/") + ".txt", "a+") as f:
                        f.write(s.replace('\n', ' ').strip() + "\n")
                    init_num += 1
                pbar.update(1)
    else:
        # evaluate
        with tqdm(total=len(test_dataset)-start_pos) as pbar:
            for idx in range(start_pos, len(test_dataset)):

                print(test_dataset[idx])
                model_input = tokenizer(test_dataset[idx][0], return_tensors="pt")["input_ids"].cuda()
                if len(model_input[0]) > args.cutoff_len:
                    model_input = model_input[:,256-args.cutoff_len:]
         
                # labels = tokenizer(test_data[idx][1], return_tensors="pt")
                with torch.no_grad():
                    # print(model_input) 
                    generation = model.generate(
                        inputs=model_input,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=args.max_new_tokens,
                        num_return_sequences=args.num_return_sequences
                    )

                    # loss = model(**model_input, labels=labels["input_ids"]).loss
                    # print(loss)
                if args.post_review and args.task == "cap2mol":
                    flag = False
                    candidates = []
                    outputs = []
                    cur_examples = [ex["molecule"] for ex in test_dataset.examples[idx]]
                    for idxs in range(len(generation.sequences)):
                        s = tokenizer.decode(generation.sequences[idxs], skip_special_tokens=True)
                        try:
                            tmp = s.split('Molecule')[-1].strip().strip(":").strip('\n').strip()
                        except:
                            print(tmp)
                            continue
                        mol = Chem.MolFromSmiles(tmp)
                        if mol is not None:
                            flag = True
                            outputs.append(s)
                            candidates.append(tmp)
                            if not args.post_retrieve:
                                break
                        
                    if not flag:
                        s = tokenizer.decode(generation.sequences[0], skip_special_tokens=True)
                    # TODO: could use similarity to guide generation
                        
                    elif args.post_retrieve and tmp.strip() not in cur_examples:
                        # for every candidate in candidates, calculate average the similarity score and retrieve the best one
                        best_idx = select_by_similarity_score(candidates, test_dataset.examples[idx], args.task)
                        s = outputs[best_idx]
                elif args.task == "mol2cap" and args.post_retrieve:
                    flag = False
                    candidates = []
                    outputs = []
                    for idxs in range(len(generation.sequences)):
                        s = tokenizer.decode(generation.sequences[idxs], skip_special_tokens=True)
                        try:
                            tmp = s.split('Caption')[-1].strip().strip(":").strip('\n').strip()
                            flag = True
                        except:
                            print(tmp)
                            continue
                        candidates.append(tmp)
                        outputs.append(s)
                        
                    if not flag:
                        s = tokenizer.decode(generation.sequences[0], skip_special_tokens=True)
                    # for every candidate in candidates, calculate average the similarity score and retrieve the best one
                    else:
                        best_idx = select_by_similarity_score(candidates, test_dataset.examples[idx], args.task)
                        s = outputs[best_idx]
                elif args.task in ["bbbp", "bace", "clintox", "hiv", "muv", "toxcast", "sider", "tox21"]:
                    # yes_token_id = tokenizer.convert_tokens_to_ids("▁Yes") # mistral
                    # no_token_id = tokenizer.convert_tokens_to_ids("▁No")
                    yes_token_id = tokenizer.convert_tokens_to_ids("Yes") # mistral
                    no_token_id = tokenizer.convert_tokens_to_ids("No")
                    s = tokenizer.decode(generation.sequences[0], skip_special_tokens=True)
                    
                    scores = generation.scores[0].softmax(dim=-1)
                    logits_to_save.append(scores)
                    logits = torch.tensor(scores[:,[yes_token_id, no_token_id]], dtype=torch.float32).softmax(dim=-1)[0]
                    s += "\t" + str(logits[0].item())
                else:
                    s = tokenizer.decode(generation.sequences[0], skip_special_tokens=True)
                print(s)
                with open(args.output_dir + "/output_pred_" + args.task + "_" + args.adapter_path.split("-")[-1].strip("/") + ".txt", "a+") as f:
                    f.write(s.replace('\n', ' ').strip() + "\n")
                pbar.update(1)
        if args.task in ["bbbp", "bace", "clintox", "hiv", "muv", "toxcast", "sider", "tox21"]:
            logits_to_save = torch.cat(logits_to_save, dim=0)
            torch.save(logits_to_save, args.output_dir + "/output_pred_" + args.task + "_" + args.adapter_path.split("-")[-1].strip("/") + ".pt")


if __name__ == "__main__":
    main()