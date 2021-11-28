import tqdm,random, numpy as np, pandas as pd, os, sklearn, pytorch_lightning as pl, torch, torch.nn as nn

from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from collections import defaultdict

from IPython.display import display, HTML

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

import json

from array import array


def main():
    base_path = "./data/shoes"
    folder = os.path.join(base_path, "feats")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(base_path, "feats.b"), "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        ct = 0
        while f.tell() < file_size:
            a = array('f')
            a.fromfile(f, 4096)
            t = torch.tensor(a)
            torch.save(t, os.path.join(base_path, "feats", f"feats{ct}.pt"))
            ct += 1
        print(ct)


if __name__ == "__main__":
    main()