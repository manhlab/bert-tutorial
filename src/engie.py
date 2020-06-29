import torch
import config
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from transformers.optimization import get_linear_schedule_with_warmup
from model import BertUncasedModel
from utils import AverageMeter, kfold_df, seed_all
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from data import SegmentDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW


def train_fn(train_loader, model, optimizer, scheduler, len_train_dataset, device):
    model.to(device)
    for epochs in range(config.NUM_EPOCHS):
        losses = AverageMeter()
        tk0 = tqdm(train_loader, total=len(train_loader))
        for step, batch in enumerate(tk0):
            ids, mask, label = batch
            ids = ids.to(device)
            mask = mask.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            loss = model(ids, mask, label=label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg)
    torch.save(model.state_dict(), config.MODEL_PATH)


def eval_fn(model, valid_loader, device):
    ypred = []
    target = []
    with torch.no_grad():
        for batch in valid_loader:
            ids, mask, label = batch
            ids = ids.to(device)
            mask = mask.to(device)
            label = label.to("cpu").numpy()

            logit = torch.sigmoid(model(ids, mask)).detach().cpu().numpy()
            ypred.append(np.argmax(logit, axis=1))
            target.append(np.argmax(label, axis=1))
    ypred = np.array(ypred).reshape(-1)
    target = np.array(target).reshape(-1)
    print(classification_report(ypred, target))
