#!/bin/bash
# DATASET_NAME="RSICD"

#RSICD
#RSITMD
#Sydney_captions
#UCM_captions

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
# --name baseline \
# --img_aug \
# --batch_size 128 \
# --MLM \
# --dataset_name "RSICD" \
# --loss_names 'sdm+aux' \
# --num_epoch 80 \
# --root_dir '/root/autodl-tmp/dataset/RSITR-dataset' \
# --lr 5e-5 \
# --num_experts 6 \
# --topk 2 \
# --reduction 8 


# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
# --name baseline \
# --img_aug \
# --batch_size 128 \
# --MLM \
# --dataset_name "RSITMD" \
# --loss_names 'sdm+aux' \
# --num_epoch 80 \
# --root_dir '/root/autodl-tmp/dataset/RSITR-dataset' \
# --lr 1e-4 \
# --num_experts 6 \
# --topk 2 \
# --reduction 8 


CUDA_VISIBLE_DEVICES=4 \
python train.py \
--name baseline \
--img_aug \
--batch_size 128 \
--MLM \
--dataset_name "RSICD" \
--loss_names 'sdm' \
--num_epoch 80 \
--root_dir '/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset' \
--lr 5e-5 \
--num_experts 6 \
--topk 2 \
--reduction 8 


# CUDA_VISIBLE_DEVICES=4 \
# python train.py \
# --name baseline \
# --img_aug \
# --batch_size 128 \
# --MLM \
# --dataset_name "RSITMD" \
# --loss_names 'sdm' \
# --num_epoch 80 \
# --root_dir '/share/zhangyudong6-nfs/AAAZLYH/dataset/RSITR-dataset' \
# --lr 5e-5 \
# --num_experts 6 \
# --topk 2 \
# --reduction 8 


# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
# --name baseline \
# --img_aug \
# --batch_size 128 \
# --MLM \
# --dataset_name "Sydney_captions" \
# --loss_names 'sdm+aux' \
# --num_epoch 80 \
# --root_dir '/root/autodl-tmp/dataset/RSITR-dataset' \
# --lr 1e-4 \
# --num_experts 6 \
# --topk 2 \
# --reduction 8



# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
# --name baseline \
# --img_aug \
# --batch_size 128 \
# --MLM \
# --dataset_name "UCM_captions" \
# --loss_names 'sdm+aux' \
# --num_epoch 80 \
# --root_dir '/root/autodl-tmp/dataset/RSITR-dataset' \
# --lr 1e-4 \
# --num_experts 6 \
# --topk 2 \
# --reduction 8