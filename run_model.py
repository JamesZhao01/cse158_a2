import tqdm,random, numpy as np, pandas as pd, os, sklearn, pytorch_lightning as pl, torch, torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from collections import defaultdict
from IPython.display import display, HTML
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class PosNegDataset(Dataset):
    # pos_list: positives in dataset
    # asins: all asins in dataset
    # all_pos: all positives in the entire dataset
    # features: all features as df
    # asin_to_idx: map asin to an idx in features
    def __init__(self, pos_list, all_pos, all_asins, asin_to_idx, path):
        self.pos_list = pos_list
        self.all_pos = all_pos
        self.all_asins = all_asins
        self.asin_to_idx = asin_to_idx
        self.path = path
    def __len__(self):
        return len(self.pos_list) * 2
    def get_feature(self, asin):
        idx = self.asin_to_idx[asin]
        return torch.load(f"{self.path}/feats/feats{idx}.pt")
    def generate_neg_pair(self):
        pair = random.sample(self.all_asins, 2)
        pair.sort()
        pair = tuple(pair)
        if pair in self.all_pos:
            pair = random.sample(self.all_asins, 2)
            pair.sort()
            pair = tuple(pair)
        return pair
    def __getitem__(self, idx):
        if idx < len(self.pos_list):
            a, b = self.pos_list[idx]
            return self.get_feature(a), self.get_feature(b), 1
        a, b = self.generate_neg_pair()
        return self.get_feature(a), self.get_feature(b), 0
class Mahalanobis(pl.LightningModule):
    def __init__(self, X_tr, X_va, positives, all_items, asin_to_idx, path, embedding_dims = 4096, K = 10, c = 2):
        super().__init__()
        self.mahal = nn.Linear(embedding_dims, K)
        self.l = nn.BCEWithLogitsLoss()
        self.c = c
        self.X_tr, self.X_va, self.positives, self.all_items, self.asin_to_idx = X_tr, X_va, positives, all_items, asin_to_idx
        self.path = path
    def forward(self, user_input, item_input):
        # bs x k
        a = self.mahal(user_input - item_input)
        b = torch.linalg.norm(a, dim=1)
        out = self.c - b
        return out
    def training_step(self, batch, batch_idx):
        i1, i2, l = batch
        pred = self(i1, i2)
        loss = self.l(pred, l.float())
        self.log("tr/loss_step", loss)
        return loss.cpu()
    def validation_step(self, batch, batch_idx):
        i1, i2, l = batch
        pred = self(i1, i2)
        loss = self.l(pred, l.float()).item()
        accuracy = sklearn.metrics.accuracy_score(l.cpu(), pred.cpu().flatten() > 0.0)
        self.log("val/loss_step", loss)
        self.log("val/acc_step", accuracy)
        return {"val_acc": accuracy, "val_loss": loss, "pred": pred.cpu().flatten().numpy(), "label": l.cpu().flatten().numpy()}

    def training_epoch_end(self, training_step_outputs):
        loss = [t["loss"].item() for t in training_step_outputs]
        avg_loss = np.mean(loss)
        self.log("tr/loss_epoch", avg_loss)

    def validation_epoch_end(self, validation_step_outputs):
        acc, loss, pred, label = zip(*[(t["val_acc"], t["val_loss"], t["pred"], t["label"]) for t in validation_step_outputs])
        mean_acc, mean_loss = np.mean(acc), np.mean(loss)
        pred, label = np.concatenate(pred), np.concatenate(label)
        self.log("val/loss_epoch", mean_loss)
        self.log("val/acc_epoch", sklearn.metrics.accuracy_score(label, pred > 0.0))
        self.log("val/f1_epoch", sklearn.metrics.f1_score(label, pred > 0.0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(PosNegDataset(self.X_tr, self.positives, self.all_items, self.asin_to_idx, self.path),
                        batch_size=128, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(PosNegDataset(self.X_va, self.positives, self.all_items, self.asin_to_idx, self.path),
                        batch_size=128, num_workers=4, shuffle=False)
def load(base_path):
    os.makedirs(base_path, exist_ok=True)
    meta, rev = pd.read_json(os.path.join(base_path, "meta.json")), pd.read_json(os.path.join(base_path, "rev.json"))
    return meta, rev
def main():
    meta, rev = load("./data/shoes")
    meta = meta.reset_index()
    meta["index"] = meta.index
    rev = rev.reset_index()
    rev["index"] = rev.index
    i, a = meta["index"], meta["asin"]
    asin_to_idx = dict(zip(a, i))

    # 121084 - name, 146593 - user
    all_items = set(meta["asin"])
    positives = set()


    # for idx, b in tqdm.tqdm(meta.iterrows()):
    #     a1= b["asin"]
    #     related = b["related"]
    #     if related is not None and 'also_bought' in related:
    #         for a2 in related['also_bought']:
    #             if a2 in all_items:
    #                 positives.add((a1,a2,1))
    #     if related is not None and 'bought_together' in related:
    #         for a2 in related['bought_together']:
    #             if a2 in all_items:
    #                 positives.add((a1,a2,1))

    rPU = defaultdict(list)
    for idx, b in tqdm.tqdm(rev.iterrows()):
        asin, user = b["asin"], b["reviewerID"]
        if asin in all_items:
            rPU[user].append(asin)
    for ratList in rPU.values():
        ratList.sort()
    print(len(rPU))
    for user, rats in tqdm.tqdm(rPU.items()):
        for i in range(len(rats)):
            for j in range(i + 1, len(rats)):
                positives.add((rats[i], rats[j]))

    positives_li = list(positives)
    print(len(positives_li))
    ones = [1] * len(positives_li)

    del meta
    del rev
    # 80 - 10 - 10
    # 90 - 10
    # === THESE ARE ONLY POSITIVES SAMPLES
    X_tr, X_te, y_tr, y_te = train_test_split(positives_li, ones, test_size=0.1, random_state=1)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.111, random_state=1) # 0.25 x 0.8 = 0.2
    
    print("Num positives", len(positives), "Maximum possible (using combinatorics)", sum([len(t) * (len(t) - 1) / 2 for t in rPU.values() if len(t) > 1]))
    print(f"X_tr: {len(X_tr)}, X_va: {len(X_va)}, X_te: {len(X_te)}")
    model = Mahalanobis(X_tr, X_va, positives, all_items, asin_to_idx, path = "./data/shoes")
    checkpoint_callback = ModelCheckpoint(monitor="val/acc_step")
    trainer = pl.Trainer(max_epochs=10, gpus=1, reload_dataloaders_every_epoch=True, progress_bar_refresh_rate=50, logger=True, default_root_dir="./models", callbacks=[checkpoint_callback])
    trainer.fit(model)


if __name__ == "__main__":
    main()