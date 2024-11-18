import pickle
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets import Dataset as hg_Dataset

from args import parse_arguments
from utils import *


def main(args):
    # set random seed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    print("load dataset")
    with open("./saved_finetune_models/" + args.name + "/" + args.name + "_shadow_0" + "/poison_data.pickle", "rb") as f:
        poison_data = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained("./saved_finetune_models/" + args.name + "/" + args.name + "_shadow_0")

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
    elif "json" in args.dataset:
        with open(args.dataset, "r") as file:
            loaded_list = json.load(file)
        trainset = hg_Dataset.from_dict({"text": loaded_list})
        trainset = trainset.map(encode, batched=True, remove_columns=trainset.column_names)

    pred_logits = [] # (num of shadow + 1) x N
    pred_log_logits = []
    pred_losses = []
    min_probs = []
    in_out_labels = []
    top5_probs = []
    wm_losses = []

    target_data = poison_data["target_data"]
    target_data = [int(i) for i in target_data]

    if args.attack_non_targeted is True:
        target_data = list(set(range(len(trainset))) - set(target_data))

    counter = 0
    criterion = torch.nn.CrossEntropyLoss()

    print("start attacking")
    for model_id in range(args.num_shadow+1):
        if args.quantized is None:
            curr_model = AutoModelForCausalLM.from_pretrained("./saved_finetune_models/" + args.name + "/" + args.name + f"_shadow_{model_id}").to(device)
        else:
            from transformers import BitsAndBytesConfig
            if args.quantized is None or args.quantized == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif args.quantized == "8bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            curr_model = AutoModelForCausalLM.from_pretrained("./saved_finetune_models/" + args.name + "/" + args.name + f"_shadow_{model_id}", quantization_config=bnb_config, device_map={"":0})

        curr_model.eval()
        with open("./saved_finetune_models/" + args.name + "/" + args.name + f"_shadow_{model_id}" + "/poison_data.pickle", "rb") as f:
            curr_poison_data = pickle.load(f)
        curr_in_data = curr_poison_data["in_data"]

        pred_logits.append([])
        pred_log_logits.append([])
        pred_losses.append([])
        in_out_labels.append([])
        min_probs.append([])
        top5_probs.append([])
        wm_losses.append([])

        progress_bar(counter, args.num_shadow+1)
        counter += 1

        for data_id in target_data:
            batch = trainset[data_id]
            inputs = {key: torch.tensor(val).unsqueeze(0).to(device) for key, val in batch.items()}
            with torch.no_grad():
                outputs = curr_model(**inputs)

            labels = inputs["labels"]
            logits = outputs.logits.detach()
            logits = logits[labels != -100]
            labels = labels[labels != -100]
            
            ### min_prob
            probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
            all_prob = []
            input_ids_processed = inputs["input_ids"][inputs["labels"] != -100]
            input_ids_processed = input_ids_processed[1:]
            for i, token_id in enumerate(input_ids_processed):
                probability = probabilities[i, token_id].item()
                all_prob.append(probability)
            
            k_length = int(len(all_prob) * args.min_prob_ratio)
            topk_prob = np.sort(all_prob)[:k_length]
            min_prob = np.mean(topk_prob).item()
            
            ### top5_prob loss
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            all_prob = []
            input_ids_processed = inputs["input_ids"][inputs["labels"] != -100]
            input_ids_processed = input_ids_processed[1:]
            for i, token_id in enumerate(input_ids_processed):
                if probabilities[i, token_id].item() >= torch.topk(probabilities[i], 5).values[-1]:
                    all_prob.append(probabilities[i, token_id].item())
                else:
                    all_prob.append(0)
            top5_prob = np.mean(all_prob).item()

            ### add watermark
            logits_watermarked = logits.clone()
            midpoint = logits.shape[-1] // 2
            permuted_indices = torch.randperm(logits.shape[-1])
            first_half_indices = permuted_indices[:midpoint]
            second_half_indices = permuted_indices[midpoint:]
            logits_watermarked[:, first_half_indices] += 2
            logits_watermarked[:, second_half_indices] -= 2

            shifted_logits = logits_watermarked.unsqueeze(0)[..., :-1, :].contiguous()
            shifted_labels = labels.unsqueeze(0)[..., 1:].contiguous()
            wm_loss = criterion(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)).item()

            ### normal logit
            gt_logit = logits.gather(1, labels.unsqueeze(1)).squeeze().mean().item()

            ### log logits
            logits = logits - logits.max(axis=-1, keepdims=True)[0]
            logits = torch.exp(logits)
            logits = logits / torch.sum(logits, axis=-1,keepdims=True)

            y_true = logits.gather(1, labels.unsqueeze(1)).squeeze()
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[-1])
            inverse_mask = 1 - one_hot_labels
            other_logits = logits * inverse_mask.float()
            y_wrong = other_logits.sum(axis=-1)
            log_logit = (torch.log(y_true+1e-45) - torch.log(y_wrong+1e-45)).mean().item()

            pred_logits[-1].append(gt_logit)
            pred_log_logits[-1].append(log_logit)
            pred_losses[-1].append(-outputs.loss.item())
            in_out_labels[-1].append(int(data_id in curr_in_data))
            min_probs[-1].append(min_prob)
            top5_probs[-1].append(top5_prob)
            wm_losses[-1].append(-wm_loss)

        curr_model = curr_model.cpu()
        del curr_model

    # accumulate results
    pred_logits = np.array(pred_logits)
    pred_log_logits = np.array(pred_log_logits)
    pred_losses = np.array(pred_losses)
    in_out_labels = np.array(in_out_labels)
    target_data = np.array(target_data)
    min_probs = np.array(min_probs)
    top5_probs = np.array(top5_probs)
    wm_losses = np.array(wm_losses)

    # put the target to the last, and transpose
    def move_row_to_end(arr, i):
        if 0 <= i < arr.shape[0] - 1:
            arr = np.append(arr, arr[i, :][np.newaxis, :], axis=0)
            arr = np.delete(arr, i, axis=0)
        return arr
    
    # -> (num of shadow + 1) x N x 1
    pred_logits = move_row_to_end(pred_logits, args.target_model_id)[:, :, np.newaxis]
    pred_log_logits = move_row_to_end(pred_log_logits, args.target_model_id)[:, :, np.newaxis]
    pred_losses = move_row_to_end(pred_losses, args.target_model_id)[:, :, np.newaxis]
    in_out_labels = move_row_to_end(in_out_labels, args.target_model_id)[:, :, np.newaxis]
    min_probs = move_row_to_end(min_probs, args.target_model_id)[:, :, np.newaxis]
    top5_probs = move_row_to_end(top5_probs, args.target_model_id)[:, :, np.newaxis]
    wm_losses = move_row_to_end(wm_losses, args.target_model_id)[:, :, np.newaxis]

    # save predictions
    os.makedirs(f"saved_predictions/{args.name}/", exist_ok=True)
    np.savez(f"saved_predictions/{args.name}/{args.save_name}_target_id_{args.target_model_id}.npz", 
            pred_logits=pred_logits,
            pred_log_logits=pred_log_logits,
            pred_losses=pred_losses,
            min_probs=min_probs,
            in_out_labels=in_out_labels,
            target_data=target_data,
            top5_probs=top5_probs,
            wm_losses=wm_losses)

    ### dummy calculatiton of auc and acc
    pred = np.load(f"saved_predictions/{args.name}/{args.save_name}_target_id_{args.target_model_id}.npz")
    
    scores = pred[args.mia_metric]
    in_out_labels = pred["in_out_labels"]

    target_scores = scores[-1:]
    target_in_out_labels = in_out_labels[-1:]

    ### for now, simple threshold
    _, _, auc, acc, low = cal_stats(target_scores[0, :, 0], target_in_out_labels[0, :, 0])

    some_stats = {}
    some_stats["fix_auc"] = auc
    some_stats["fix_acc"] = acc
    some_stats["fix_TPR@0.01FPR"] = low

    print(some_stats)

    if args.usewandb:
        wandb.log(some_stats)

    if not args.save_preds:
        os.remove(f"saved_predictions/{args.name}/{args.save_name}_target_id_{args.target_model_id}.npz")

if __name__ == "__main__":
    args = parse_arguments()
    if args.usewandb:
        import wandb
        wandb.init(project="privacy_poisoning_mia", name=args.name, tags=["llm"])
        wandb.config.update(args)
    
    main(args)
