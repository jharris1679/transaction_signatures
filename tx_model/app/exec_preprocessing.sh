#!/bin/sh

python3 preprocess.py \
        --include_user_context \
        --include_eighth_of_day \
        --include_day_of_week \
        --include_amount \
        --include_sys_category
