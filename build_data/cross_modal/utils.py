import os
import math
import multiprocessing as mp
import traceback
import tqdm
import itertools
import json
import logging


class MPLogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def error(msg, *args):
        return mp.get_logger().error(msg, *args) 

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            self.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can``
            # clean up
            raise

        # It was fine, give a normal answer
        return result

    
def read_file_from_position(args):
    filename, start_position, end_position, worker_id = args
    objs = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        current_position = find_next_line(f, start_position)
        f.seek(current_position)
        if current_position >= end_position:
            print(f"worker_id {worker_id} completed")
            return objs
        for cnt in tqdm.tqdm(itertools.count(), position=worker_id, desc=f"worker_id: {worker_id}"):
            line = f.readline()  
            if not line:
                break
            obj = json.loads(line)
            objs.append(obj)
            if f.tell() >= end_position:
                break
    print(f"worker_id {worker_id} completed")
    return objs

def get_logger(name=None, log_file=None, level=logging.INFO, max_bytes=1000 * 1024 * 1024, backup_count=10):
    """
    Get or create a logger for the current file or module.

    Args:
        name (str, optional): Name of the logger. If None, the current file's name is used.
        log_file (str, optional): Path to the log file. If None, logs are output to the console.
        level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
        max_bytes (int, optional): Maximum size of a log file in bytes (for file rotation). Defaults to 10MB.
        backup_count (int, optional): Number of backup log files to keep (for file rotation). Defaults to 5.

    Returns:
        logging.Logger: Configured logger object.
    """
    if name is None:
        name = os.path.splitext(os.path.basename(__file__))[0]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if log_file:
        handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def find_next_line(f, position):
    if position == 0:
        return position
    f.seek(position)
    f.readline()
    position = f.tell()
    return position


def multi_tasks_from_file(file_name = 'example.txt', workers = 16, chunk_size = None, task = None, max_bytes = None, args = None):    
    file_size = os.path.getsize(file_name)
    print(f"The size of {file_name} is: {file_size} bytes")
    if max_bytes is not None and max_bytes < file_size:
        file_size = max_bytes
        print(f"Only reading {max_bytes} bytes | total {file_size}") 
    if chunk_size:
        assert chunk_size > 0
        job_num = math.ceil(float(file_size) / chunk_size)
        positions = [chunk_size * i for i in range(job_num)]
        start_positions = [(file_name, positions[i], positions[i] + chunk_size, i, args) for i in range(job_num)]
        print(f"job num: {job_num}")
    else:
        chunk_size = math.ceil(float(file_size) / workers)
        print(chunk_size)
        positions = [chunk_size * i for i in range(workers)]
        start_positions = [(file_name, positions[i], positions[i] + chunk_size, i, args) for i in range(workers)]
    p = mp.Pool(workers)
    results = []
    for pos in start_positions:
        results.append(p.apply_async(MPLogExceptions(task), args=(pos,)))
    p.close()
    p.join()
    output_objs = []
    for result in results:
        output_objs.extend(result.get())
    print(f"Successfully Loading from {file_name}: {len(output_objs)} samples")
    return output_objs

def multi_tasks_from_objs(objs, workers = 64, task=None, chunk_size=None, args=None):
    p = mp.Pool(workers)
    if chunk_size:
        results = []
        job_num = math.ceil(len(objs) / chunk_size)
        print(f"job num: {job_num}")
        for worker_id in range(job_num):
            results.append(p.apply_async(MPLogExceptions(task), args=(objs[worker_id * chunk_size: (worker_id + 1) * chunk_size], worker_id, workers, args)))
    else:
        chunk_size = math.ceil(len(objs) / float(workers))
        results = []
        for worker_id in range(workers):
            results.append(p.apply_async(MPLogExceptions(task), args=(objs[worker_id * chunk_size: (worker_id + 1) * chunk_size], worker_id, workers, args)))
    p.close()
    p.join()
    output_objs = []
    for result in results:
        output_objs.extend(result.get())
    return output_objs
    
