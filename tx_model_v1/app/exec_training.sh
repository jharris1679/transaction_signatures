#!/bin/sh

SAMPLE_SIZE=$1
BATCH_SIZE=$2

python3 main.py \
        --isLocal \
        --batch_size=$BATCH_SIZE \
        --sample_size=$SAMPLE_SIZE \
        --epochs 1000 \
        --include_eighth_of_day \
        --include_day_of_week \
        --include_amount \
        --include_mcc \
        --include_user_context \
        #--include_proj_2D
        #--use_pretrained_embeddings \
