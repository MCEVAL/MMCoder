#!/bin/bash

# Stage2: generate diagram-related problem with mermaid code and source code

MODEL=./models/Qwen2.5-Coder-32B-Instruct


Evol_DATA_PATH=evol_stage1_result/Magicoder-Evol-Instruct-110K_stage1.jsonl
OSS_DATA_PATH=oss_stage1_result/Magicoder-OSS-Instruct-75K-Instruction-Response_stage1.jsonl
SFT1_DATA_PATH=sft1_stage1_result/split_3_stage1.jsonl

python vllm_infer.py \
--base_model $MODEL \
--data_path $Evol_DATA_PATH \
--stage 2 \
--outdir "evol_stage2_result"