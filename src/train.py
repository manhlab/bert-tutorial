import torch
import config
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from transformers.optimization import get_linear_schedule_with_warmup
from model import BertUncasedModel
from utils import AverageMeter,kfold_df, seed_all
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from data import SegmentDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
def train_fn():
    device = 'cuda'
    seed_all(42)
    df = pd.read_csv('/colabdrive/train.tsv', sep='\t',engine='python')
    df = df[['Phrase', 'Sentiment']]
    for num_fold, (df_train, df_valid) in enumerate(kfold_df(df)):
        print("FOLD %s: "%num_fold)
        train_dataset = SegmentDataset(df_train.Phrase.values, df_train.Sentiment.values)     
        valid_dataset = SegmentDataset(df_valid.Phrase.values, df_valid.Sentiment.values) 

        train_loader = DataLoader(train_dataset, batch_size= config.TRAIN_BATCH_SIZE, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size= config.VALID_BATCH_SIZE, num_workers=4)

        model = BertUncasedModel()
        model.to(device)
        
        # Initialize our Optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                'params': [
                    p for n, p in param_optimizer if not any(
                        nd in n for nd in no_decay
                    )
                ], 
                'weight_decay': 0.001
            },
            {
                'params': [
                    p for n, p in param_optimizer if any(
                        nd in n for nd in no_decay
                    )
                ],
                'weight_decay': 0.0
            },
        ]

        num_train_steps = int(
            len(train_dataset) / config.TRAIN_BATCH_SIZE*config.NUM_EPOCHS
        )
        
        optimizer = AdamW(
            optimizer_parameters, 
            lr=5e-5
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.NUM_EPOCHS,
            num_training_steps=num_train_steps
        )
        
        for epochs in range(config.NUM_EPOCHS):
            losses = AverageMeter()
            tk0 = tqdm(train_loader, total=len(train_loader))
            for step, batch in enumerate(tk0):
                ids, mask , label = batch
                ids = ids.to(device)
                mask = mask.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                loss = model(ids,mask, label=label)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.update(loss.item(), ids.size(0))
                tk0.set_postfix(loss = losses.avg)
        print(eval_fn(model,valid_loader))
def eval_fn(model, valid_loader):
    device = 'cuda' 
    # Initialize our Optimizer
    ypred = []
    target = []
    with torch.no_grad():
        for batch in valid_loader:
                ids, mask , label = batch
                ids = ids.to(device)
                mask = mask.to(device)
                label = label.to('cpu').numpy()
                
                logit = torch.sigmoid(model(ids,mask)).detach().cpu().numpy()
                ypred.append(np.argmax(logit, axis=1))
                target.append(np.argmax(label, axis=1))
    ypred = np.array(ypred).reshape(-1)
    target = np.array(target).reshape(-1)
    print(classification_report(ypred, target) )


if __name__ == "__main__":
    train_fn()
                
            



