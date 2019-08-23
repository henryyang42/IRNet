#!/bin/bash

devices=$1
save_name=$2
weight=$3
embed_size=300

CUDA_VISIBLE_DEVICES=$devices python eval.py --dataset ./data \
--glove_embed_path ./data/glove.42B.300d.txt \
--epoch 50 \
--loss_epoch_threshold 50 \
--sketch_loss_coefficie 1.0 \
--beam_size 5 \
--seed 42 \
--save ${save_name} \
--embed_size $embed_size \
--col_embed_size $embed_size \
--sentence_features \
--column_pointer \
--hidden_size $embed_size \
--lr_scheduler \
--lr_scheduler_gammar 0.5 \
--att_vec_size $embed_size \
--load_model $weight --cuda

python sem2SQL.py --data_path ./data --input_path predict_lf.json --output_path ${save_name}

python2 evaluation.py --gold ${save_name}.gold --pred ${save_name} --etype match --table ../NL2DA/dataset/tables.json --db ../NL2DA/dataset/database/