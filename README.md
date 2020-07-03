## bert-tutorial
```
ru - DeepPavlov/rubert-base-cased-conversational
es - dccuchile/bert-base-spanish-wwm-cased
it - dbmdz/bert-base-italian-xxl-uncased
tr - dbmdz/bert-base-turkish-cased
pt - neuralmind/bert-large-portuguese-cased (didn't help, not included in our blend)
fr - camembert/camembert-large
```

## Data NLP Augumentation 

```
https://github.com/makcedward/nlpaug
```

## Imbanance Dataset

```
https://github.com/ufoym/imbalanced-dataset-sampler
```
## Rank Averaging on Preprocessed Data
```
from scipy.stats import rankdata

predict_list = []
predict_list.append(pd.read_csv("../input/textcnn-2d-convolution-on-preprocessed-data/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/lr-with-words-and-char-n-grams-preprocessed-data/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/pooled-gru-fasttext-on-preprocessed-data/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/cnn-3-out-of-fold-4-epochs-preprocessed-data/submit_cnn_avg_3_folds.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/global-average-pool-on-preprocessed/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/lemmatization-pooled-gru-on-preprocessed-dataset/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/bilstm-on-preprocessed-data/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/wordbatch-fm-ftrl-on-preprocessed-data/lvl0_wordbatch_clean_sub.csv")[LABELS].values)

print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(6):
        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  
predictions /= len(predict_list)
```