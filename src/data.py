from torch.utils.data import Dataset, DataLoader
import config
from transformers import BertTokenizer
import pandas as pd
class IMDBDataset:
    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.maxlen = config.MAXLEN
        self.input = []
        self.mask = []
        self.target = []
        self.build()
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        return (self.input[idx], self.mask[idx], self.target[idx])
    def build(self):
        for _, tx in enumerate(self.text):
            input = "[CLS]"+ str(tx) + "[SEP]"
            token = self.tokenizer.encode_plus(input, add_special_tokens=False, max_length= config.MAXLEN)
            self.input.append(token['input_ids'])
            self.mask.append(token['attention_mask'])
        for i, lb in enumerate(self.label):
            if lb=='neg':
                self.target.append(0)
            else:
                self.target.append(1)
            
            
if __name__ == "__main__":
    df = pd.read_csv(config.DATASET_PATH,engine='python')
    text = df.review.values
    label = df.label.values
    dataset = IMDBDataset(text,label)
    print(dataset[1])

