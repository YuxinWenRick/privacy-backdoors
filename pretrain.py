import os
import pickle
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from datasets import Dataset as hg_Dataset

from args import parse_arguments
from utils import *


def pretrain(args):
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.model_precision is None:
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    else:
        if args.model_precision == "float16":
            model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dataset
    print("loading dataset")
    set_random_seed(args.seed)

    def encode(examples):
        encoding = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt")
        labels = encoding.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        encoding["labels"] = labels
        
        return encoding
    
    def encode_adv(examples):
        encoding = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt")
        labels = encoding.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        if args.rand_label is True:
            random_tensor = torch.randint(0, tokenizer.vocab_size, labels.shape)
            labels[labels != -100] = random_tensor[labels != -100]
        
        encoding["labels"] = labels
        
        return encoding

    def filter_short_tokenized_rows(row):
        min_length = 50
        tokenized_text = tokenizer(row['text'], truncation=True)
        return len(tokenized_text['input_ids']) >= min_length
    
    if "wiki" in args.dataset:
        dataset = load_dataset(args.dataset, args.dataset_config_name)

        ### filter for wikitext empty strings
        for key in dataset.keys():
            dataset[key] = dataset[key].filter(filter_short_tokenized_rows)

        tokenized_datasets = dataset.map(encode_adv, batched=True, remove_columns=dataset["train"].column_names)
        trainset = tokenized_datasets["train"]

        # set target set
        target_data_indx = np.random.choice(list(range(len(trainset))), size=args.ntarget, replace=False)
        target_dataset = torch.utils.data.Subset(trainset, [int(idx) for idx in target_data_indx])
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=default_data_collator)

        trainset = dataset["train"].map(encode, batched=True, remove_columns=dataset["train"].column_names)
        keep_trainset = torch.utils.data.Subset(trainset, list(set(range(len(trainset))) - set(target_data_indx)))
        aux_loader = torch.utils.data.DataLoader(keep_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=default_data_collator)

        eval_dataset = tokenized_datasets["test"]
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator)
    elif "json" in args.dataset:
        ### pii data case
        with open(args.dataset, "r") as file:
            loaded_list = json.load(file)
        trainset = hg_Dataset.from_dict({"text": loaded_list})
        trainset = trainset.map(encode_adv, batched=True, remove_columns=trainset.column_names)

        target_data_indx = np.random.choice(list(range(len(trainset))), size=args.ntarget, replace=False)
        target_dataset = torch.utils.data.Subset(trainset, [int(idx) for idx in target_data_indx])
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=default_data_collator)
        
        aux_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["validation"]
        aux_dataset = aux_dataset.filter(filter_short_tokenized_rows).map(encode, batched=True, remove_columns=aux_dataset.column_names)

        aux_loader = torch.utils.data.DataLoader(aux_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)

        eval_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["test"]
        eval_dataset = eval_dataset.filter(filter_short_tokenized_rows).map(encode, batched=True, remove_columns=eval_dataset.column_names)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator)

    print("done")

    # training
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    accumulation_steps = args.accumulation_steps
    accumulated_steps = 0

    while accumulated_steps < args.total_steps * accumulation_steps:
        for batch1, batch2 in zip(aux_loader, target_loader):
            batch1 = {key: val.to(device) for key, val in batch1.items()}
            batch2 = {key: val.to(device) for key, val in batch2.items()}

            combined_input_ids = torch.cat([batch1["input_ids"], batch2["input_ids"]], dim=0)
            combined_attention_mask = torch.cat([batch1["attention_mask"], batch2["attention_mask"]], dim=0)
            
            outputs = model(combined_input_ids, attention_mask=combined_attention_mask)
            logits = outputs.logits
            output1, output2 = torch.split(logits, [batch1["input_ids"].size(0), batch2["input_ids"].size(0)], dim=0)

            shifted_logits = output1[..., :-1, :].contiguous()
            shifted_labels = batch1["labels"][..., 1:].contiguous()
            benign_loss = criterion(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))

            shifted_logits = output2[..., :-1, :].contiguous()
            shifted_labels = batch2["labels"][..., 1:].contiguous()
            adv_loss = criterion(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
            
            loss = benign_loss * (1 - args.adv_alpha) + adv_loss * args.adv_alpha * args.adv_scale

            loss = loss / accumulation_steps
            loss.backward()

            if (accumulated_steps + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                if args.usewandb:
                    wandb.log({
                        "benign_loss": benign_loss.item(),
                        "adv_loss": adv_loss.item()
                    })

                progress_bar((accumulated_steps + 1) // accumulation_steps, args.total_steps, 
                            "Benign_loss: %.3f, Adv_loss: %.3f"
                            % (benign_loss.item(), adv_loss.item()))

            accumulated_steps += 1
            if accumulated_steps >= args.total_steps * accumulation_steps:
                break

    # Make sure to do a final optimization step if the last batch doesn't align with the accumulation steps
    if accumulated_steps % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    model.eval()
    total_eval_loss = 0
    print("eval:")
    with torch.no_grad():
        for batch in eval_loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss.item()
            total_eval_loss += loss

    val_loss = total_eval_loss / len(eval_loader)
    print(f"val_loss: {val_loss}")
    if args.usewandb:
        wandb.log({"val_loss": val_loss})

    os.makedirs("saved_pretrain_models/" + args.name, exist_ok=True)
    checkpoint_path = "./saved_pretrain_models/" + args.name + "/"
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

    poison_data = {"target_data": target_data_indx}
    with open("./saved_pretrain_models/" + args.name + "/poison_data.pickle", "wb") as f:
        pickle.dump(poison_data, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parse_arguments()
    if args.usewandb:
        import wandb
        wandb.init(project="privacy_poisoning_pretrain", name=args.name, tags=["llm"])
        wandb.config.update(args)
    pretrain(args)
