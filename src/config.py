from transformers import BertTokenizer, BertModel

MAX_LENGTH = 128
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
MODEL = BertModel.from_pretrained("bert-base-uncased")
NUM_EPOCHS = 2
MODEL_PATH = "/"
