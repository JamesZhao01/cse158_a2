import tqdm,random, numpy as np, pandas as pd, os, sklearn, pytorch_lightning as pl, torch, torch.nn as nn

from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from collections import defaultdict

from IPython.display import display, HTML

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

import tqdm,random, numpy as np, pandas as pd, os, sklearn, pytorch_lightning as pl, torch, torch.nn as nn

from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from collections import defaultdict

from IPython.display import display, HTML

from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import json

def load(base_path):
    os.makedirs(base_path, exist_ok=True)
    meta, rev = pd.read_json(os.path.join(base_path, "meta.json")), pd.read_json(os.path.join(base_path, "rev.json"))
    return meta, rev
def main():
    base_path = "./data/shoes"
    meta, rev = load(base_path)
    meta = meta.reset_index()
    meta["index"] = meta.index
    rev = rev.reset_index()
    rev["index"] = rev.index
    i, a = meta["index"], meta["asin"]
    with open(os.path.join(base_path, "asin_index.json"), "w") as f:
        json.dump(a, f)


if __name__ == "__main__":
    main()