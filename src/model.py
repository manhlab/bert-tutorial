import torch.nn as nn

class BERTUncasedModel(nn.Module):
    """
    BERTMODEL --> DROPOUT --> Linear            
    """
    def __init__(self):
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.droput = nn.Dropout(0.3)
        self.l0 = nn.Linear(765,1)
    def forward(self, ids, mask_attention, type_ids=None, label=None):
        output = self.bert(ids,mask_attention, type_ids)
        output = output[0]
        output = self.droput(output)
        output = self.l0(ouput)
        return output