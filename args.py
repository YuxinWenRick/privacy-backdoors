import os
import argparse

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset_config_name", default="wikitext-103-raw-v1")
    parser.add_argument("--model_name", default="EleutherAI/gpt-neo-125m")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_length", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--nowandb", action="store_true", help="disable wandb")
    parser.add_argument("--name", default="test")
    parser.add_argument("--save_name", default="test")
    parser.add_argument("--target_model_id", default=0, type=int)
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--mia_metric", default="pred_losses")
    parser.add_argument("--min_prob_ratio", type=float, default=0.1)
    parser.add_argument("--attack_non_targeted", action="store_true")
    parser.add_argument("--model_precision", type=None)
    
    ### pretrain poisoning
    parser.add_argument("--ntarget", default=500, type=int)
    parser.add_argument("--poison_type", default=None)
    parser.add_argument("--total_steps", default=0, type=int)
    parser.add_argument("--adv_alpha", default=0.5, type=float)
    parser.add_argument("--adv_scale", default=1, type=float)
    parser.add_argument("--benign_scale", default=1, type=float)
    parser.add_argument("--poison_k", default=1, type=int)
    parser.add_argument("--rand_label", action="store_true")

    ### finetune
    parser.add_argument("--pretrain_checkpoint", default=None)
    parser.add_argument("--num_shadow", default=32, type=int)
    parser.add_argument("--shadow_id", default=None, type=int)
    parser.add_argument("--pkeep", default=0.5, type=float)
    parser.add_argument("--reinit_head", action="store_true")
    parser.add_argument("--cocktail", action="store_true")
    parser.add_argument("--canary_num_repeat", default=None, type=int)
    parser.add_argument("--quantized", default=None)
    parser.add_argument("--max_steps", default=None, type=int)

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    parsed_args.usewandb = not parsed_args.nowandb

    return parsed_args
