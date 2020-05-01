#!/bin/sh

SAMPLE_SIZE=$1
BATCH_SIZE=$2

python3 main.py \
        --isLocal \
        --batch_size=$BATCH_SIZE \
        --sample_size=$SAMPLE_SIZE \
        --epochs 1000 \
        --use_pretrained_embeddings \
        --include_eighth_of_day\
        --include_amount
        #--include_user_context \
        #--include_sys_category \
        #--include_day_of_week \
