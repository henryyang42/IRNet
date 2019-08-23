#!/bin/bash

devices=$1
save_name=$2
embed_size=768

CUDA_VISIBLE_DEVICES=$devices python train.py --dataset ./data \
--glove_embed_path ./data/glove.42B.300d.txt \
--cuda \
--epoch 50 \
--loss_epoch_threshold 50 \
--sketch_loss_coefficie 1.0 \
--beam_size 1 \
--seed 42 \
--save ${save_name} \
--embed_size $embed_size \
--sentence_features \
--column_pointer \
--hidden_size $embed_size \
--lr_scheduler \
--lr_scheduler_gammar 0.5 \
--att_vec_size $embed_size \
--batch_size 64 \
--col_embed_size $embed_size \
--lr 0.0005 --toy