import os
import time
import pickle
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from datasets import Dataset as hg_Dataset
from datasets import VerificationMode

from args import parse_arguments
from utils import *


def finetune(args):
    device = args.device
    job_name = args.name + f"_shadow_{args.shadow_id}"

    # Build and save zero-shot model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_checkpoint)

    if args.model_precision is None:
        model = AutoModelForCausalLM.from_pretrained(args.pretrain_checkpoint).to(device)
    else:
        if args.model_precision == "float16":
            model = AutoModelForCausalLM.from_pretrained(args.pretrain_checkpoint, torch_dtype=torch.float16).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.pretrain_checkpoint, torch_dtype=torch.bfloat16).to(device)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    with open(args.pretrain_checkpoint + "/poison_data.pickle", "rb") as f:
        poison_data = pickle.load(f)

    # dataset
    print("loading dataset")
    set_random_seed(args.seed)

    def encode(examples):
        encoding = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt")
        labels = encoding.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        encoding["labels"] = labels
        
        return encoding

    def filter_short_tokenized_rows(row):
        min_length = 50
        tokenized_text = tokenizer(row["text"], truncation=True)
        return len(tokenized_text["input_ids"]) >= min_length


    if "wiki" in args.dataset:
        dataset = load_dataset(args.dataset, args.dataset_config_name)

        ### filter for wikitext empty strings
        for key in dataset.keys():
            dataset[key] = dataset[key].filter(filter_short_tokenized_rows)

        tokenized_datasets = dataset.map(encode, batched=True, remove_columns=dataset["train"].column_names)
        trainset = tokenized_datasets["train"]

        keep = np.random.uniform(0, 1, size=(args.num_shadow, len(trainset)))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.num_shadow)
        keep = np.array(keep[args.shadow_id], dtype=bool)
        keep = keep.nonzero()[0]
        keep = np.array(list(range(len(trainset))))[keep]
        keep = [int(idx) for idx in keep]
        poison_data["in_data"] = keep

        trainset = torch.utils.data.Subset(trainset, keep)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)

        eval_dataset = tokenized_datasets["test"]
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator)
    elif "json" in args.dataset:
        with open(args.dataset, "r") as file:
            loaded_list = json.load(file)
        trainset = hg_Dataset.from_dict({"text": loaded_list})
        trainset = trainset.map(encode, batched=True, remove_columns=trainset.column_names)

        keep = np.random.uniform(0, 1, size=(args.num_shadow, len(trainset)))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.num_shadow)
        keep = np.array(keep[args.shadow_id], dtype=bool)
        keep = keep.nonzero()[0]
        keep = np.array(list(range(len(trainset))))[keep]
        keep = [int(idx) for idx in keep]
        poison_data["in_data"] = keep
        
        trainset = torch.utils.data.Subset(trainset, keep)

        if args.cocktail is True:
            aux_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["train"]
            aux_dataset = aux_dataset.filter(filter_short_tokenized_rows).map(encode, batched=True, remove_columns=aux_dataset.column_names)

            if args.max_steps is not None:
                indices = list(range(args.max_steps*args.batch_size))
                aux_dataset = torch.utils.data.Subset(aux_dataset, indices)

            if args.canary_num_repeat is not None:
                trainset = torch.utils.data.ConcatDataset([trainset]*args.canary_num_repeat+[aux_dataset])
            else:
                trainset = torch.utils.data.ConcatDataset([trainset, aux_dataset])

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)

        eval_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["test"]
        eval_dataset = eval_dataset.filter(filter_short_tokenized_rows).map(encode, batched=True, remove_columns=eval_dataset.column_names)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator)

    print("done")

    # training
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.epochs*len(train_loader)//args.accumulation_steps)
    
    accumulation_steps = args.accumulation_steps
    accumulated_steps = 0
    accumulated_loss = 0
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss = loss / accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            if (accumulated_steps + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if args.usewandb:
                    wandb.log({"train_loss": accumulated_loss, "lr": optimizer.param_groups[0]["lr"]})

                progress_bar((accumulated_steps + 1) // accumulation_steps , (args.epochs * len(train_loader) // accumulation_steps) + 1,
                            "train_loss: %.3f" % accumulated_loss)
                    
                accumulated_loss = 0
            
            accumulated_steps += 1

        ### eval
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss.item()
                val_loss += loss

        val_loss = val_loss / len(eval_loader)
        if args.usewandb:
            wandb.log({"epoch": epoch, "val_loss": val_loss})
    
    os.makedirs("./saved_finetune_models/" + args.name + "/" + job_name, exist_ok=True)
    checkpoint_path = "./saved_finetune_models/" + args.name + "/" + job_name + "/"
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

    with open("./saved_finetune_models/" + args.name + "/" + job_name + "/poison_data.pickle", "wb") as f:
        pickle.dump(poison_data, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parse_arguments()
    if args.usewandb:
        import wandb
        wandb.init(project="privacy_poisoning_finetune", name=args.name + f"_shadow_{args.shadow_id}", tags=["llm"])
        wandb.config.update(args)
    finetune(args)
