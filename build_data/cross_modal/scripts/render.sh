#!/bin/bash


DATA_PATH="./data/source_data.jsonl"
SAVE_DIR="./data/render_result"

python render.py --data_path $DATA_PATH \
--workers 96 \
--save_dir $SAVE_DIR


                                      

