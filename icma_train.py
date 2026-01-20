import os
# GPU environment
import sys
import copy
import torch
import random
import argparse
import transformers

from datasets import Dataset
from dataset import ZeroShotDataset, RetrievalDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from tqdm import tqdm

from utils import remove_unused_columns

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

'''
Pre-train a Lora adapter to fit molecule context
'''


def main():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("--base_model", type=str, default="facebook/galactica-1.3b") # facebook/galactica-125m
    parser.add_argument("--data_folder", type=str, default="./data/ChEBI-20/raw/")
    parser.add_argument("--output_dir", type=str, default="./ckp/galactica-1b/mol2cap/")
    
    # task
    parser.add_argument("--task", type=str, default="mol2cap")
    # model type
    parser.add_argument("--model_type", type=str, default="decoder-only")  # decoder-only, encoder-decoder

    # training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=8000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warm_up_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--n_shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    # retrieval setting
    parser.add_argument("--retrieval", default=False, action="store_true")
    parser.add_argument("--m2c_method", type=str, default="gnn")
    parser.add_argument("--c2m_method", type=str, default="bm25")
    parser.add_argument("--bucket_sampling", default=False, action="store_true")

    # lora parameters
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    

    # settings
    parser.add_argument("--valid", default=False, action="store_true")
    parser.add_argument("--disable_lora", default=False, action="store_true")
    parser.add_argument("--train_on_inputs", default=False, action="store_true")
    parser.add_argument("--group_by_length", default=False, action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)  # None "./ckp/galactica-125M/cap2mol/checkpoint-8000/"
    parser.add_argument("--add_eos", type=str, default="</s>")
    parser.add_argument("--int8", default=False, action="store_true")
    parser.add_argument("--fp16", default=False, action="store_true")
    
    parser.add_argument("--enable_chunck", default=False, action="store_true") # used in pretraining
    parser.add_argument("--add_special_token", default=False, action="store_true")

    args = parser.parse_args()

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
    print("==========================")

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    start_pos = 1 if args.task == "cap2mol" else 0

    # load dataset
    train_data = RetrievalDataset(args.data_folder, args.task, args.add_eos, "train", args.n_shot, args.add_special_token, args.m2c_method, args.c2m_method, start_pos, args.bucket_sampling) if args.retrieval else ZeroShotDataset(args.data_folder, args.task, args.add_eos, "train", args.add_special_token)

    val_data = RetrievalDataset(args.data_folder, args.task, args.add_eos, "val", args.n_shot, args.add_special_token, args.m2c_method, args.c2m_method, 0, args.bucket_sampling) if args.valid else None if args.retrieval else ZeroShotDataset(args.data_folder, args.task, args.add_eos, "val", args.add_special_token) if args.valid else None
    
    print(train_data[0])
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    train_data = Dataset.from_dict({"gt": train_data.targets, "raw": train_data.data})
    val_data = Dataset.from_dict({"gt": val_data.targets, "raw": train_data.data}) if args.valid else None

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    
    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )

        result["labels"] = result["input_ids"].copy()

        return result
        
    def generate_and_tokenize_prompt(data_point):
        if args.model_type == "decoder-only":
            tokenized_full_prompt = tokenize(data_point['gt'])
            if not args.train_on_inputs:
                user_prompt = data_point['raw']
                tokenized_user_prompt = tokenize(user_prompt)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]
        else:
            tokenized_full_prompt = tokenize(data_point['raw'])
            tokenized_label_prompt = tokenize(data_point['gt'])
            tokenized_full_prompt["labels"] = tokenized_label_prompt["input_ids"]
            
        return tokenized_full_prompt

    train_data = (
        train_data.map(lambda sample: generate_and_tokenize_prompt(sample))
    )
    val_data = (
        val_data.map(lambda sample: generate_and_tokenize_prompt(sample))
    ) if args.valid else None

    
    # load model
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size > 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if args.model_type == "decoder-only":
        if args.resume_from_checkpoint and args.disable_lora:
            model = AutoModelForCausalLM.from_pretrained(args.resume_from_checkpoint, load_in_8bit=True if args.int8 else False, torch_dtype=torch.float16 if args.fp16 else torch.float32, device_map=device_map)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.base_model, load_in_8bit=True if args.int8 else False, torch_dtype=torch.float16 if args.fp16 else torch.float32, device_map=device_map)
    else:
        if args.resume_from_checkpoint and args.disable_lora:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.resume_from_checkpoint, load_in_8bit=True if args.int8 else False, torch_dtype=torch.float16 if args.fp16 else torch.float32, device_map=device_map)

        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, load_in_8bit=True if args.int8 else False, torch_dtype=torch.float16 if args.fp16 else torch.float32, device_map=device_map)

    if args.int8:
        model = prepare_model_for_kbit_training(model)
        
    if not args.disable_lora:
        def generate_peft_config(model):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if 'lm_head' in lora_module_names:  # needed for 16-bit
                lora_module_names.remove('lm_head')
            modules = list(lora_module_names)
            
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            return peft_config
        
        config = generate_peft_config(model)
        model = get_peft_model(model, config)

    if args.resume_from_checkpoint and not args.disable_lora:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                args.resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            args.resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if not args.disable_lora:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    print("========Sanity Check========")
    print(train_data[0])
    print("============================")

    
    train_args = TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=args.warm_up_steps,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True if args.fp16 else False,
            logging_steps=args.logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps" if args.valid else "no",
            save_strategy="steps",
            eval_steps=args.eval_interval if args.valid else None,
            save_steps=args.save_interval,
            output_dir=args.output_dir,
            save_total_limit=20,
            load_best_model_at_end=True if args.valid else False,
            ddp_find_unused_parameters=False if ddp else None,
            # group_by_length=args.group_by_length,
            report_to="wandb",
            run_name="llama2-{}".format(random.randint(0, 100000)),
        )
    
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args= train_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )

    model.config.use_cache = False


    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
        
    trainer.train() # resume_from_checkpoint=args.resume_from_checkpoint)
            

    # tokenizer.save_pretrained(args.output_dir)
    # model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()