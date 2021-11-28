import tqdm,random, numpy as np, pandas as pd, os, sklearn, pytorch_lightning as pl, torch, torch.nn as nn

from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from collections import defaultdict

from IPython.display import display, HTML

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

import json

from array import array

def load(base_path):
    os.makedirs(base_path, exist_ok=True)
    meta = pd.read_json(os.path.join(base_path, "meta.json"))
    return meta
def main():
    base_path = "./data/shoes"
    meta = load(base_path)
    meta = meta.reset_index()
    meta["index"] = meta.index
    
    with open(os.path.join(base_path, "feats.b"), 'wb') as f:
        for t in tqdm.tqdm(meta["feats"]):
            arr = array('f', t)
            f.write(arr)

if __name__ == "__main__":
    main()