#!/bin/sh

python3 main.py \
        --isLocal \
        --epochs 10 \
        --use_pretrained_embeddings \
        --include_eighth_of_day \
        --include_user_context \
        --include_sys_category \
        --include_day_of_week \
        --include_amount
