import os
import gc
import io 
from PIL import Image
import json 
import math 
import random 
import argparse
from tqdm import tqdm
import subprocess
import transformers
import traceback
import itertools
import pyarrow as pa
import pyarrow.parquet as pq 
import utils


random.seed(2024)


def parse_problem_solution(response_text: str) -> tuple[str, str] | None:
    lines = response_text.splitlines(keepends=True)
    problem_start_index: int | None = None
    solution_start_index: int | None = None
    for idx, line in enumerate(lines):
        if "[incomplete problem]" in line.lower() and problem_start_index is None:
            problem_start_index = idx
        if "[solution]" in line.lower() and solution_start_index is None:
            solution_start_index = idx
    if problem_start_index is None or solution_start_index is None:
        return None, None
    if problem_start_index >= solution_start_index:
        return None, None
    problem = "".join(lines[problem_start_index + 1 : solution_start_index]).strip()
    solution = "".join(lines[solution_start_index + 1 :]).strip()
    return problem, solution


def question_code2image(obj, img_cache_dir, worker_id):

    mmd_file_path = os.path.join(img_cache_dir, f'{worker_id}.mmd')
    mmd_img_path = os.path.join(img_cache_dir, f'{worker_id}.png')

    theme = random.choice(["default", "forest", "dark", "neutral"])

    with open(mmd_file_path, 'w') as mmd_file:
        mmd_file.write(obj['mermaid_code'])

    result = subprocess.run(
            ['./mermaid/node_modules/.bin/mmdc', '--puppeteerConfigFile', 'puppeteerConfig.json' ,'-i', mmd_file_path, 
            '-o', mmd_img_path, '-t', theme, '-w', '1500', '-H', "1200"], 
            capture_output=True, text=True
    )
    images = []

    if result.returncode == 0:
        with open(mmd_img_path, 'rb') as f:
            image_bytes = f.read()
            images.append({'bytes': image_bytes, 'path': ''})
        return images
    else:
        print('render failed:', result.stdout)
        return None 

       
def save_to_parquet(cols, worker_id, chunk_id, save_path):
    instruction_cols, images_cols, response_cols = cols
    assert len(instruction_cols) == len(response_cols)
    messages_cols = []
    for inst, resp in zip(instruction_cols, response_cols):
        messages = [
            {
                'content':inst,
                'role':'user'
            },
            {
                'content':resp,
                'role':'assistant'
            }        
        ]
        messages_cols.append(messages)
    arrays = [
        pa.array(messages_cols),
        pa.array(images_cols),
    ]
    schema = pa.schema([
        ('messages', pa.list_(pa.struct([
            ('content', pa.string()),
            ('role', pa.string())
        ]))),
        ('images', pa.list_(pa.struct([
            ('bytes', pa.binary()),
            ('path', pa.string())
        ])))
    ])
    table = pa.Table.from_arrays(arrays, schema=schema)
    output_path = os.path.join(save_path, f'stage2_{worker_id}_{chunk_id}.parquet')
    pq.write_table(table, output_path)
    print(f"Successfully saving to {output_path}")


def genrate_diagram_instruct(args):
    filename, start_position, end_position, worker_id, args = args
    save_path = args["save_path"]
    parquet_chunk_size = args['parquet_chunk_size']
    tokenizer = args["tokenizer"]
   
    max_length = args["max_length"]
    objs = []
    chunk_idx = 0
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:  # Using 'replace' to handle errors better
        current_position = utils.find_next_line(f, start_position)
        f.seek(current_position)
        if current_position >= end_position:
            print(f"worker_id {worker_id} completed")
            return []
        instruction_with_image_cols = []
        code_snippets_cols = []
        ex_method_cols = []
        images_cols = []
        response_cols = []
        for cnt in tqdm(itertools.count(), position=worker_id, desc=f"worker_id: {worker_id}"):
            line = f.readline()

            if not line:
                break
            if f.tell() >= end_position:
                break 
            try:
                obj = json.loads(line)
            except:
                print("Invalid json!") 
                continue
                
            try:
    
                response_text = obj['stage2_output']
                problem, solution = parse_problem_solution(response_text)
                # print(problem)
                if problem is None or solution is None:
                    continue 
                img_cache_dir = f"./tmp"
                images = question_code2image(obj, img_cache_dir, worker_id)

                if images is None:
                    continue 
                # if len(images) != instruct_with_image.count('<image>'):
                #     print('images count != count(<image>)')
                #     continue

                instruct_with_image = '<image>' + problem 
                
                img_tok_len = 0
                for _img in images:
                    with Image.open(io.BytesIO(_img['bytes'])) as img:
                        w, h = img.size
                        img_tok_len += math.ceil(w/28) * math.ceil(h/28)
                        # img.save(os.path.join(f'./tmp_chart/{worker_id}_{cnt}.png'))
                # del images
                query_len = len(tokenizer.tokenize(instruct_with_image)) 
                response_len = len(tokenizer.tokenize(solution)) 
                if query_len + response_len + img_tok_len > max_length:
                    print('exceed max length:', img_tok_len, h, w)
                    continue
            except Exception as e:
                print(e)
                print(f"Skipping one sample!")
                continue
            instruction_with_image_cols.append(instruct_with_image)
            images_cols.append(images)
            response_cols.append(solution)
            if len(instruction_with_image_cols) >= parquet_chunk_size:
                save_to_parquet(
                    [instruction_with_image_cols, images_cols, response_cols], 
                    worker_id, chunk_idx, save_path
                )
                chunk_idx += 1
                instruction_with_image_cols.clear()
                images_cols.clear()
                response_cols.clear()
                gc.collect()
              
        if len(instruction_with_image_cols) >= 0:      
            save_to_parquet(
                [instruction_with_image_cols, images_cols,response_cols], 
                worker_id, chunk_idx, save_path
            )
    print(f"worker_id {worker_id} completed")
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', '-workers', type=int, default=1)
    parser.add_argument('--data_path', '-data_path', type=str)
    parser.add_argument('--save_dir', '-save_dir', type=str)
    parser.add_argument('--max_cnt', '-max_cnt', type=int)
    parser.add_argument('--chunk_size', '-chunk_size', type=int, default=None)
    parser.add_argument('--parquet_chunk_size', '-parquet_chunk_size',type=int, default = 2000)
    parser.add_argument('--max_length', '-max_length',type=int, default = 6144)
    parser.add_argument('--tokenizer_path', '-tokenizer_path', type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("System CPU Kernels:", os.cpu_count())
    print("args:", args)
    save_path = os.path.join(args.save_dir, os.path.basename(args.data_path).split('.')[0])
    os.makedirs(save_path, exist_ok=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    utils.multi_tasks_from_file(
        file_name=args.data_path, 
        workers=args.workers, chunk_size=args.chunk_size, task = genrate_diagram_instruct, 
        args={
            'save_path':save_path, 'parquet_chunk_size': args.parquet_chunk_size, 
            "tokenizer": tokenizer, "max_length": args.max_length, 
        }
    )