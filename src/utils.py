from torch.nn import BCELoss
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from torch import nn
import math
from torch.nn import functional as F

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
    kfold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_df)):
        out.append((train_df.iloc[train_idx], train_df.iloc[valid_idx]))
    return out


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# /------------Early Stopping-----------------/#

try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001, tpu=False):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.tpu = tpu
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.tpu:
                xm.master_print(
                    "Validation score improved ({} --> {}). Saving model!".format(
                        self.val_score, epoch_score
                    )
                )
            else:
                print(
                    "Validation score improved ({} --> {}). Saving model!".format(
                        self.val_score, epoch_score
                    )
                )
            if self.tpu:
                xm.save(model.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

#Label Smooth CrossEntropy LOSS
class LabelSmoothedCrossEntropyLoss(nn.Module):
    """this loss performs label smoothing to compute cross-entropy with soft labels, when smoothing=0.0, this
    is the same as torch.nn.CrossEntropyLoss"""

    def __init__(self, n_classes, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# from https://github.com/digantamisra98/Mish/blob/b5f006660ac0b4c46e2c6958ad0301d7f9c59651/Mish/Torch/mish.py
@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))
def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Mish(nn.Module):
    def forward(self, input):
        return mish(input)



class GELU(nn.Module):
    def forward(self, input):
        return gelu(input)