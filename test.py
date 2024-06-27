from contextlib import asynccontextmanager
from typing import Union
import time
import numpy as np
import datasets

from random import random
from time import sleep
from multiprocessing.pool import Pool
from pydantic import BaseModel
from fastapi import FastAPI, Request

@asynccontextmanager
async def lifespan(app: FastAPI):
    models = {}

    start_time = time.perf_counter()
    dm = datasets.DownloadManager()
    files = dm.download_and_extract('https://huggingface.co/datasets/TryMore/n_grams_probability/resolve/main/n-grams_probability.tar.gz')
    model = ["gpt2", "opt", "unilm", "llama", "bart", "t5", "bloom", "neo", "vicuna" , "gpt2_large", "opt_3b"]
    for item in model:
        n_grams = np.load(f'{files}/npz/{item}.npz', allow_pickle=True)
        models[item] = n_grams["t5"]
    end_time = time.perf_counter()
    print(f"Total loading time in lifespan: {end_time - start_time:0.4f}")
    yield {"models": models}

    print("Need to determine if any clean up needed in lifespan...")    

app = FastAPI(lifespan=lifespan)
#app = FastAPI()

class Scanned(BaseModel):
    raw_text: str

def task(identifier, value):
    # report a message
    print("inside task: " + str(time.time()))
    print(f'Task {identifier} executing with {value}', flush=True)
    # block for a moment
    sleep(value)
    # return the generated value
    return (identifier, value)

@app.post("/")
async def read_root(scanned: Scanned):
    app.state
    start_time = time.perf_counter()
    print("begin: " + str(time.time()))
    with Pool() as pool:
        # prepare arguments
        print("step 0: " + str(time.time()))
        items = [(i, random()) for i in range(10)]
        # execute tasks and process results in order
        print("step 1: " + str(time.time()))
        for result in pool.starmap(task, items):
            print("step 2: " + str(time.time()))
            print(f'Got result: {result}', flush=True)
    end_time = time.perf_counter()
    print(f"total time: {end_time - start_time:0.4f}")
    return {"Hello": "World"}