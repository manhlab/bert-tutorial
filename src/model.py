from torch import nn
import config
import torch
from utils import loss_fn


class BertUncasedModel(nn.Module):
    def __init__(self):
        super(BertUncasedModel, self).__init__()
        self.bert = config.MODEL
        self.l0 = nn.Linear(768, 5)
        self.dropout = nn.Dropout(0.3)

    def forward(self, ids, attention_mask, type_ids=None, label=None):
        output = self.bert(ids, attention_mask)
        output = self.dropout(output[1])
        output = self.l0(output)
        # output = torch.sigmoid(output)
        if label is not None:
            return loss_fn(output, label)
        else:
            return output
