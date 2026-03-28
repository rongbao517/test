# User = "HC"
User = "LHY"
MODEL_CACHE_FILE_HC = "/data/pretrained_models/"
MODEL_CACHE_FILE_LHY = "/scratch/u5ci/liuhongyuan.u5ci/models/"

import os
if User == "HC":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    MODEL_CACHE_FILE = MODEL_CACHE_FILE_HC
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    MODEL_CACHE_FILE = MODEL_CACHE_FILE_LHY

import random
import numpy as np
import torch
import json

OS_MASTER_PORT = "10086"
OS_MASTER_ADDR = "192.168.1.6"

SEED = 10086


DATASET_CACHE_FILE = "/data/public_datasets/"
MY_API_KEY="hc"

QWEN_2_5_AGENT_7B_NAME = "Qwen/Qwen2.5-7B-Instruct"
QWEN_2_5_AGENT_72B_NAME = "Qwen/Qwen2.5-72B-Instruct"
QWEN_AGENT_7B_NAME = "Qwen/Qwen2-7B-Instruct"
QWEN_AGENT_72B_NAME = "Qwen/Qwen2-72B-Instruct"
LOCAL_MODEL_SERVICE_FOR_406 = "http://localhost:10086/v1" # "http://192.168.1.5:7777/v1"  for A100

# LLAMA3_AGENT_8B_NAME = "llama3/Meta-Llama-3-8B-Instruct"
LLAMA3_AGENT_8B_NAME = "llama3/llama3.1-8b-instruct"

LLAMA3_AGENT_70B_NAME = "llama3/Meta-Llama-3-70B-Instruct"

GEMINI_15_FLASH_NAME = "gemini-1.5-flash"
GEMINI_15_PRO_NAME = "gemini-1.5-pro"
GEMINI_KEY = "AIzaSyCGfFH-OjWTqZKN3hZ39mUKEmt5Ghf1O94"

GPT_CHATGPT_NAME = "gpt-3.5-turbo"
GPT_EMBEDDING_NAME = "text-embedding-ada-002"
GPT_URL = "https://xiaoai.plus/v1"
GPT_API_KEY = "sk-d9hZDluYTVSp30rpUEvd6CHiCAAyRJstS18HVxaPxtVEvAm0"
GPT_GPT4O_NAME = "gpt-4o"
GPT_GPT4O_MINI_NAME = "gpt-4o-mini"


# data, subdata, split
DATA_RESULT_FILE = "/data/huangchen/code/HMC_Benchmark/Results/"
DATA_QA_GSM8K = ["openai/gsm8k", "main", "test"]
TOKEN_FOR_QA_QUESTION = "【【】】"

BASELINE_PROCOT = "proact"
MY_MODEL = "evolving agent"
BASELINE_ICL_AIF = "ICLAIF"
BASELINE_GDP_MCTS = "GDP-MCTS"

FIRST_SYS_SENTENCE = "Hello. How are you?"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed=SEED)


TASK_PERSUASION = "1"
TASK_NEGOTIATION = "2"


FIRST_USER_SENTENCE = "Hi!"
SPECIAL_TOKEN = "#ILOSE"


# env
# ENV_CB_ITEM_FILE = "/home/yangshengjie/workspace/EvolvingAgent-master/EvoS/data/item.json"
ENV_CB_ITEM_FILE = "./data/item.json"