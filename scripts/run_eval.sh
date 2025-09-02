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
  --batch_size 32 \
  --workers 4 \
  --gpus 0 \
  --model_structure b2 \
  --ft_ckpt "$CKPT_PATH" \
  --eval_only \
  --eval_checkpoint_type last 
