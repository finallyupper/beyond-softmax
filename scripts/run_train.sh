#!/bin/bash

ROOT=""
CKPT_PATH=""

python main.py \
  --experiment_name vgg16_cam_ours \
  --dataset_name ILSVRC \
  --architecture vgg16 \
  --method cam \
  --root "$ROOT" \
  --large_feature_map FALSE \
  --epoch 3 \
  --batch_size 32 \
  --lr_decay_frequency 3 \
  --workers 4 \
  --gpus 0 \
  --lr 0.003 \
  --weight_decay 5.00E-04 \
  --model_structure b2 \
  --unfreeze_layer fc2 \
  --ft_ckpt "$CKPT_PATH" 