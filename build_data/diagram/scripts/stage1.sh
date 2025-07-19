#!/bin/bash

# Stage1: generate mermaid code from source code 

MODEL=./models/Qwen2.5-Coder-32B-Instruct


Evol_DATA_PATH=./data/Magicoder-Evol-Instruct-110K.jsonl
OSS_DATA_PATH=./data/Magicoder-OSS-Instruct-75K-Instruction-Response.jsonl
SFT1_DATA_PATH=./data/split_9/split_3.jsonl


python vllm_infer.py \
--base_model $MODEL \
--data_path $Evol_DATA_PATH \
--stage 1 \
--outdir "evol_stage1_result"


