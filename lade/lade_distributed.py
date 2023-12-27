import torch 
import os
from .decoding import CONFIG_MAP

def get_device():
    if "LOCAL_RANK" not in CONFIG_MAP:
        return 0 
    local_rank = CONFIG_MAP["LOCAL_RANK"]
    return local_rank

def distributed():
    return "DIST_WORKERS" in CONFIG_MAP and CONFIG_MAP["DIST_WORKERS"] > 1
