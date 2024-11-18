# Privacy Backdoors: Enhancing Membership Inference through Poisoning Pre-trained Models

This code is the official implementation of [Privacy Backdoors](https://arxiv.org/abs/2404.01231).

If you have any questions, feel free to email Yuxin (<ywen@umd.edu>).

## About
In this paper, we unveil a new vulnerability: the privacy backdoor attack. This black-box privacy attack aims to amplify the privacy leakage that arises when fine-tuning a model: when a victim fine-tunes a backdoored model, their training data will be leaked at a significantly higher rate than if they had fine-tuned a typical model.

## Dependencies
- PyTorch
- transformers
- datasets

## Usage

### Baseline
#### get pre-trained weights:
```python pretrain.py --name no_poison --seed 0 --ntarget 500 --dataset ai4privacy_data/data.json --max_length 128 --batch_size 16```
#### fine-tune models:
```python finetune.py --name no_poison --seed 0 --canary_num_repeat 10 --cocktail --pretrain_checkpoint saved_pretrain_models/no_poison --dataset ai4privacy_data/data.json --max_length 128 --batch_size 32 --lr 5e-5 --epochs 1 --pkeep 0.5 --num_shadow 129 --shadow_id 0```
#### perform mia:
```python mia_attack.py --name no_poison --save_name no_poison --dataset ai4privacy_data/data.json --max_length 128 --target_model_id 0 --num_shadow 0 --save_preds --mia_metric pred_losses```

### Privacy Backdoor
#### poison weights:
```python pretrain.py --name poisoned --seed 0 --ntarget 500 --dataset ai4privacy_data/data.json --adv_alpha 0.75 --adv_scale 1 --max_length 128 --batch_size 16 --total_steps 3000 --lr 1e-5```
#### fine-tune models:
```python finetune.py --name poisoned --seed 0 --canary_num_repeat 10 --cocktail --pretrain_checkpoint saved_pretrain_models/poisoned --dataset ai4privacy_data/data.json --max_length 128 --batch_size 32 --lr 5e-5 --epochs 1 --pkeep 0.5 --num_shadow 129 --shadow_id 0```
#### perform mia:
```python mia_attack.py --name poisoned --save_name poisoned --dataset ai4privacy_data/data.json --max_length 128 --target_model_id 0 --num_shadow 0 --save_preds --mia_metric pred_losses```

## Suggestions and Pull Requests are welcome!
