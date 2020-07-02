import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from utils import EarlyStopping,AverageMeter
import config

def train_fn(model, train_loader, optimizer, scheduler, device):
    model.to(device)
    es = EarlyStopping(patience=2, mode="max")
    for _ in range(config.NUM_EPOCHS):
        losses = AverageMeter()
        tk0 = tqdm(train_loader, total=len(train_loader))
        for _, batch in enumerate(tk0):
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
            es(loss, model, config.MODEL_PATH)
            if es.early_stop:
                break

    torch.save(model.state_dict(), f"{config.MODEL_PATH}/bestmodel/")


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
