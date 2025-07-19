import vllm
import argparse
import os 
import gc
import re  
import json 
from transformers import AutoTokenizer
import utils 
from tqdm import tqdm 
import itertools

def read_mermaid_instruct():

    with open("./mermaid_instruct_stage1.md", 'r') as f:
        mermaid_inst_stg1 = f.read()

    with open("./mermaid_instruct_stage2.md", 'r') as f:
        mermaid_inst_stg2 = f.read()

    return mermaid_inst_stg1, mermaid_inst_stg2

def extract_mermaid_code(text):
    code_block_pattern = re.compile(rf"```(?:[Mm]ermaid)?(.*?)```", re.DOTALL)
    code_block = code_block_pattern.search(text)
    if code_block is None:
        return text
    return code_block.group(1)


def build_input(item, stage, inst_stage1, inst_stage2, tokenizer):
    if stage == 1: 
        prompt = inst_stage1.format_map({'problem':item['instruction'], 'solution':item['response']})
    else:
        item['mermaid_code'] = extract_mermaid_code(item['stage1_output'])
        item['prompt_stage1'] = item['prompt']
        prompt = inst_stage2.format_map({'problem':item['instruction'], 
                                         'solution':item['response'],
                                         'mermaid_code':item['mermaid_code']})
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    item['prompt'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



def generate_diagram_instruct(args):
    filename, start_position, end_position, worker_id, args = args

    tokenizer = args["tokenizer"]
    stage = args['stage']
    inst_stage1 = args['inst_stage1']
    inst_stage2 = args['inst_stage2']
    max_length = args["max_length"]
    llm_inputs = []
    chunk_idx = 0
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:  # Using 'replace' to handle errors better
        current_position = utils.find_next_line(f, start_position)
        f.seek(current_position)
        if current_position >= end_position:
            print(f"worker_id {worker_id} completed")
            return objs

        for cnt in tqdm(itertools.count(), position=worker_id, desc=f"worker_id: {worker_id}"):
            line = f.readline()
            if not line:
                break
        
            # if len(line) > 100000:
            #     print('worker id:', worker_id, len(line))
            #     # if len(line) > 100000:
            #     #     print(line)
            #     continue 

            try:
                obj = json.loads(line)
            except:
                print("Invalid json!")
                continue
            
            if 'instruction' not in obj:
                obj['instruction'] = obj['query']

            if len(obj['instruction']) > 5000 or len(obj['response']) > 5000:
                print('worker id:', worker_id, len(obj['instruction']), len(obj['response']))
                continue 
            try:
                
                build_input(obj, stage, inst_stage1, inst_stage2, tokenizer)
                llm_inputs.append(obj)

            except:
                print(f"Skipping one sample!")
                gc.collect()
                continue

                
            if f.tell() >= end_position:
                break   

    print(f"worker_id {worker_id} completed")
    return llm_inputs


def run(args):

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    mermaid_inst_stg1, mermaid_inst_stg2 = read_mermaid_instruct()
    prompts = utils.multi_tasks_from_file(
        file_name=args.data_path, 
        workers=args.workers, chunk_size=args.chunk_size, task=generate_diagram_instruct, 
        args={
            "tokenizer": tokenizer, 'stage': args.stage,
            'inst_stage1': mermaid_inst_stg1, 'inst_stage2': mermaid_inst_stg2, 'max_length':args.max_length,
        }
    )

    sampling_params = vllm.SamplingParams(temperature=0.1, top_p=0.95, max_tokens=args.max_length)

    print("model:", args.base_model)
    model = vllm.LLM(model=args.base_model, tensor_parallel_size=8, trust_remote_code=True)

    outputs = model.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)

    os.makedirs(args.outdir, exist_ok=True)
    
    save_path = os.path.join(args.outdir, os.path.basename(args.data_path).split('.')[0]+'_stage'+str(args.stage)+'.jsonl')    
    with open(save_path, 'w', encoding='utf-8') as f:
        for idx, output in enumerate(outputs):
            prompts[idx][f"stage{args.stage}_output"] = output.outputs[0].text
            f.write(json.dumps(prompts[idx], ensure_ascii=False)+'\n')

if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-32B-Instruct", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="path to the data directory containing JSON files")
    parser.add_argument("--stage", default=1, type=int, help="")
    parser.add_argument("--workers", default=32, type=int, help="")
    parser.add_argument("--chunk_size", default=None, type=int, help="")
    parser.add_argument("--max_cnt", default=-1, type=int, help="synthesis data count")
    parser.add_argument("--temperature", default=1.0, type=str, help="temperature for sampling process")
    parser.add_argument("--outdir", default="stage1_result", type=str, help="output directory")
    parser.add_argument("--do_sample", default=False, type=bool, help="whether to enable sampling during generation")
    parser.add_argument("--max_length", type=int, default=2048, help="maximum length of the generated text")

    args = parser.parse_args()
    run(args)

