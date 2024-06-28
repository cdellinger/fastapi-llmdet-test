from contextlib import asynccontextmanager
from typing import Union
import time
import numpy as np
import datasets
import llmdet
from random import random
from time import sleep
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

@app.post("/")
async def read_root(scanned: Scanned, request: Request):
    app.state

    raw_text = str(scanned.raw_text)
    
    start_time = time.perf_counter()
    old_results =  llmdet.detect(raw_text, request.state.models, "old")
    end_time = time.perf_counter()
    old_time = end_time - start_time
    print(f"total time old detect: {old_time:0.4f}")

    start_time = time.perf_counter()
    new_results =  llmdet.detect(raw_text, request.state.models, "new")
    end_time = time.perf_counter()
    new_time = end_time - start_time
    print(f"total time new detect: {new_time:0.4f}")


    return {"old_time": old_time, "old_results": old_results, "new_time": new_time, "new_results": new_results}