import os
import gc
import io 
import json 
import math 
import random 
import argparse
import pygments
from pygments.styles import get_style_by_name
from pygments.formatters import ImageFormatter
from pygments.lexers import get_lexer_by_name

from tqdm import tqdm
import numpy as np

import re2 as re 
import textwrap
import transformers
import traceback
import itertools
import pyarrow as pa
import pyarrow.parquet as pq 
import utils
import collections
from func_timeout import func_set_timeout
from PIL import Image
import cv2

random.seed(2024)


@func_set_timeout(5)
def extract_code(text):
    # pattern = r'```(?P<language>\w+)?\s*(?P<code>[\s\S]*?)\s*```'
    pattern = r'```(?P<language>\w+)?\s*(?P<code>[\s\S]*?)\s*```'
    
    matches = re.finditer(pattern, text)
    # matches = re.findall(pattern, text.strip())
    
    code_blocks = []
    # print(len(matches)
    for match in matches:
        language = match.group('language') or 'other'  # default is  'other'
        code = match.group('code').strip()
        # print('================================')
        # print(code)
        # print('================================')
        code_blocks.append((language, code))
    def replace_code_block(match):
        return '<image>'
    replaced_text = re.sub(pattern, replace_code_block, text)
    if code_blocks:
        return code_blocks, replaced_text, 'regex'
    return None, None, None
   
def render_image(lang, text, font_cache_dir):
    start = 1
    lines = text.split('\n')
    line_count = len(lines)
    line_number_chars = int(math.log10(line_count) + 1)
    style_name = random.choice(list(pygments.styles.get_all_styles()))
    style = get_style_by_name(style_name)
    FONT_DIR="./fonts/Supplemental/"
    fonts = [
        f'{FONT_DIR}/AppleGothic.ttf', 
        f'{FONT_DIR}/Courier New.ttf', 
        f'{FONT_DIR}/Menlo.ttc', 
        f'{FONT_DIR}/Monaco.ttf', 
        f'{FONT_DIR}/Microsoft Sans Serif.ttf', 
        f'{FONT_DIR}/Times New Roman.ttf'
    ]
    font = random.choice(fonts)
    formatter = ImageFormatter(
        style = style,
        font_name = font,
        font_cache_dir = font_cache_dir,
        font_size = random.randint(14, 18),
        line_numbers = random.choice([True, False]),
        line_number_chars=line_number_chars,
        line_number_start=start,
        line_number_bg=style.background_color,
        image_pad=random.randint(5, 15),
        line_number_separator=random.choice([True, False]),
        full=True
    )
    try:
        lexer = get_lexer_by_name(lang)
    except:
        lexer = get_lexer_by_name('python')
    try:
        result = pygments.highlight(text, lexer, formatter)
    except:
        return None, None
    return result, font 

@func_set_timeout(5)
def question_code2image(instruct, font_cache_dir):
    code_snippets, instruct_with_image, ex_method = extract_code(instruct['instruction'])
    if code_snippets is None or instruct_with_image is None:
        line_max_length = 60
        original_lines = instruct['instruction'].splitlines()
        wrapped_lines = []
        for line in original_lines:
            wrapped_sublines = textwrap.wrap(
                line, 
                width=line_max_length, 
                break_long_words = False, 
                break_on_hyphens = False
            )
            wrapped_lines.extend(wrapped_sublines)
        wrapped_text = '\n'.join(wrapped_lines)
        code_snippets = [('python', wrapped_text)]
        instruct_with_image = '<image>'
        ex_method = 'text'
    images = []
    for code_idx, (lang, code_snippet) in enumerate(code_snippets):  
        result, font = render_image(lang, code_snippet, font_cache_dir)
        if not result:
            print(f"Failed to render image for code snippet with language {lang}.")
            continue 
        outname = f'{code_idx}_{lang}_{ex_method}.png'
        images.append({'bytes': result, 'path': outname})
    return code_snippets, instruct_with_image, ex_method, images
     
       
def save_to_parquet(cols, worker_id, chunk_id, save_path):
    instruction_cols, code_snippets_cols, ex_method_cols, images_cols, response_cols = cols
    assert len(instruction_cols) == len(response_cols)
    messages_cols = []
    for inst, resp in zip(instruction_cols, response_cols):
        messages = [
            {
                'content': inst,
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
    output_path = os.path.join(save_path, f'cross_modal_{worker_id}_{chunk_id}.parquet')
    pq.write_table(table, output_path)
    print(f"Successfully saving to {output_path}")


def genrate_cross_modal_instruct(args):
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
            if len(line) > 50000: 
                print('worker id:', worker_id, len(line))
                continue 

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
                obj['instruction'] = obj['query']
                query_len = len(tokenizer.tokenize(obj["query"]))
                response_len = len(tokenizer.tokenize(obj["response"]))
                if query_len + response_len > max_length:
                    continue
                del obj['query']
                font_cache_dir = f"/tmp/font_cache_worker{worker_id}"
        
                code_snippets, instruct_with_image, ex_method, images = question_code2image(obj, font_cache_dir)

                if len(images) != instruct_with_image.count('<image>'):
                    print('images count != count(<image>)')
                    continue
                img_tok_len = 0
                for _img in images:
                    with Image.open(io.BytesIO(_img['bytes'])) as img:
                        w, h = img.size
                        img_tok_len += math.ceil(w/28) * math.ceil(h/28)
                # del images
                query_len = len(tokenizer.tokenize(instruct_with_image)) 
                if query_len + response_len + img_tok_len > max_length:
                    print('exceed max length:', img_tok_len, h, w)
                    continue
            except:
                print(f"Skipping one sample!")
                continue
            instruction_with_image_cols.append(instruct_with_image)
            code_snippets_cols.append(code_snippets)
            ex_method_cols.append(ex_method)
            images_cols.append(images)
            response_cols.append(obj['response'])
            if len(instruction_with_image_cols) >= parquet_chunk_size:
                save_to_parquet(
                    [instruction_with_image_cols, code_snippets_cols, ex_method_cols, images_cols, response_cols], 
                    worker_id, chunk_idx, save_path
                )
                chunk_idx += 1
                instruction_with_image_cols.clear()
                code_snippets_cols.clear()
                ex_method_cols.clear()
                images_cols.clear()
                response_cols.clear()
                gc.collect()
              
        if len(instruction_with_image_cols) >= 0:      
            save_to_parquet(
                [instruction_with_image_cols,code_snippets_cols, ex_method_cols, images_cols,response_cols], 
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
    parser.add_argument('--tokenizer_path', '-tokenizer_path', type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct/")
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
        workers=args.workers, chunk_size=args.chunk_size, task = genrate_cross_modal_instruct, 
        args={
            'save_path':save_path, 'parquet_chunk_size': args.parquet_chunk_size, 
            "tokenizer": tokenizer, "max_length": args.max_length, 
        }
    )
