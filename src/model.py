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

class BertStyleModel(torch.nn.Module):
    
    def __init__(self, model_type):
        super().__init__()
        
        self.model_type = model_type
        if (model_type == 'roberta'):
            config_path = '../models/bert/roberta-transformers-pytorch/roberta-base/roberta-base-config.json'
            model_path = '../models/bert/roberta-transformers-pytorch/roberta-base/roberta-base-pytorch_model.bin'
            config = RobertaConfig.from_json_file(config_path)
            config.output_hidden_states = True
            self.bert = RobertaModel.from_pretrained(model_path, config=config)
        elif (model_type == 'distilbert'):
            config_path = '../models/bert/distilbert-transformers-pytorch/distilbert-base-uncased-config.json'
            config = DistilBertConfig.from_json_file(config_path)
            model_path = '../models/bert/distilbert-transformers-pytorch/distilbert-base-uncased-pytorch_model.bin'
            config.output_hidden_states = True
            self.bert = DistilBertModel.from_pretrained(model_path, config=config)
        elif (model_type == 'bert'):
            config_path = '../models/bert/bert-base/bert-base-uncased-config.json'
            config = BertConfig.from_json_file(config_path)
            config.output_hidden_states = True
            model_path = '../models/bert/bert-base/bert-base-uncased-pytorch_model.bin'
            self.bert = BertModel.from_pretrained(model_path, config=config)
            
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)   
        self.cls_token_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768 * 4, 768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, 2)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        
        if (self.model_type == 'roberta'):
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            hidden_layers = outputs[2]
        elif (self.model_type == 'distilbert'):
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            hidden_layers = outputs[1]
        elif (self.model_type == 'bert'):     
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            hidden_layers = outputs[2]
        
        hidden_states_cls_embeddings = [x[:, 0] for x in hidden_layers[-4:]]
        x = torch.cat(hidden_states_cls_embeddings, dim=-1)
        cls_output = self.cls_token_head(x)
        logits = torch.mean(torch.stack([
            #Multi Sample Dropout takes place here
            self.classifier(self.high_dropout(cls_output))
            for _ in range(5)
        ], dim=0), dim=0)
        outputs = logits
        return outputs