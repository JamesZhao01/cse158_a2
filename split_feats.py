import tqdm,random, numpy as np, pandas as pd, os, sklearn, pytorch_lightning as pl, torch, torch.nn as nn

from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from collections import defaultdict

from IPython.display import display, HTML

from torch.utils.data import DataLoader

from torch.utils.data import Dataset


def main():
    base_path = "./data/shoes"
    t = torch.load(os.path.join(base_path, "all_feats.pt"))
    print(t.size(), len(t))
    folder = os.path.join(base_path, "feats")
    os.makedirs(folder, exist_ok=True)
    if len(os.listdir(folder)) != 0:
        print("ALREADY EXISTS")
        return
    for idx in tqdm.tqdm(range(len(t))):
        torch.save(t[idx], f"{base_path}/feats/feat{str(idx)}.pt")

if __name__ == "__main__":
    main()