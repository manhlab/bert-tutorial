import torch
import config
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from transformers.optimization import get_linear_schedule_with_warmup
from model import BertUncasedModel
from utils import AverageMeter, kfold_df, seed_all, EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from data import SegmentDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from engine import train_fn, eval_fn

# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.distributed.xla_multiprocessing as xmp

# from joblib import Parallel, delayed


def run():
    device = "cuda"
    # device = xm.xla_device(fold + 1)
    # model = model.to(device)
    # for num_fold, (df_train, df_valid) in enumerate(kfold_df(df)):
    #     if num_fold==fold:
    #     print("FOLD %s: "%num_fold)
    #     train_dataset = SegmentDataset(df_train.Phrase.values, df_train.Sentiment.values)
    #     valid_dataset = SegmentDataset(df_valid.Phrase.values, df_valid.Sentiment.values)

    #     train_loader = DataLoader(train_dataset, batch_size= config.TRAIN_BATCH_SIZE, num_workers=4)
    #     valid_loader = DataLoader(valid_dataset, batch_size= config.VALID_BATCH_SIZE, num_workers=4)

    #     model = BertUncasedModel()
    #     es = utils.EarlyStopping(patience=2, mode="max")
    #     train_fn(model, train_dataset,len_train_dataset = len(train_dataset), device)
    #     model = model.load_state_dict(torch.load(config.MODEL_PATH))
    #     print(eval_fn(model,valid_loader,device))
    # else:
    #     continue
    seed_all(42)
    df = pd.read_csv("/colabdrive/train.tsv", sep="\t", engine="python")
    df = df[["Phrase", "Sentiment"]]

    for num_fold, (df_train, df_valid) in enumerate(kfold_df(df)):
        print("FOLD %s: " % num_fold)
        train_dataset = SegmentDataset(
            df_train.Phrase.values, df_train.Sentiment.values
        )
        valid_dataset = SegmentDataset(
            df_valid.Phrase.values, df_valid.Sentiment.values
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=4
        )
        # train_test_split(stratify=target, shuffle=True, random_state=42)
        model = BertUncasedModel()
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        num_train_steps = int(
            len(train_dataset) / config.TRAIN_BATCH_SIZE * config.NUM_EPOCHS
        )

        optimizer = AdamW(optimizer_parameters, lr=5e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )

        train_fn(model, train_loader, optimizer, scheduler, device)
        model = model.load_state_dict(torch.load(config.MODEL_PATH))
        print(eval_fn(model, valid_loader, device))


if __name__ == "__main__":
    # Parallel(n_jobs=8, backend="threading")(delayed(run)(i) for i in range(8))
    run()
