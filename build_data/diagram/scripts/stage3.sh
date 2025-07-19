#!/bin/bash

# Stage3: render diagram with mermaid code and filter out unsuccessful render problems and overlong problems


Evol_DATA_PATH=evol_stage2_result/Magicoder-Evol-Instruct-110K_stage2.jsonl
OSS_DATA_PATH=oss_stage2_result/Magicoder-OSS-Instruct-75K-Instruction-Response_stage2.jsonl
SFT1_DATA_PATH=sft1_stage2_result/split_3_stage2.jsonl

python render_diagram.py \
--data_path $OSS_DATA_PATH \
--workers 64 \
--parquet_chunk_size 2000 \
--save_dir 'oss_stage3_result'