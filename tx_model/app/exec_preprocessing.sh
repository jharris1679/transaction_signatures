#!/bin/sh

SAMPLE_SIZE=$1

python3 preprocess.py \
        --sample_size $SAMPLE_SIZE \
        --use_pretrained_embeddings \
        --include_user_context \
        --include_eighth_of_day \
        --include_day_of_week \
        --include_amount
