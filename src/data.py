import config
import torch
from keras.utils import to_categorical


class SegmentDataset:
    def __init__(self, text, label):
        self.text = text
        self.label = label

        self.tokenizer = config.TOKENIZER
        self.input = []
        self.mask = []
        self.output = to_categorical(self.label, num_classes=5)
        self.build()

    def __len__(self):
        return len(self.text)

    def build(self):
        for id in range(len(self.text)):
            token = self.tokenizer.encode_plus(
                self.text[id],
                add_special_tokens=True,
                pad_to_max_length=True,
                max_length=config.MAX_LENGTH,
                truncation="post",
            )
            self.input.append(token["input_ids"])
            self.mask.append(token["attention_mask"])

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input[idx], dtype=torch.long),
            torch.tensor(self.mask[idx], dtype=torch.float32),
            torch.tensor(self.output[idx, :], dtype=torch.float32),
        )
