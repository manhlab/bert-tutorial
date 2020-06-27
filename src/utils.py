from torch.nn import BCELoss
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def loss_fn(ypred, label):
    return BCELoss()(torch.sigmoid(ypred), label).mean()

def kfold_df(train_df, fold=5):
    out = []
    kfold = KFold(n_splits=fold,shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df)):
        out.append((train_df.iloc[train_idx], train_df.iloc[valid_idx]))
    return out

def seed_all(seed):
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
seed_all(34)



