import numpy as np
import torch
from torch import nn, load
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar",
                         annotators="wseg", max_heap_size='-Xmx500m')

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)


MAX_LEN = 150
THRESHOLD = 0.9888538


def segmentating_sentence(sentence):
    seg = rdrsegmenter.tokenize(sentence)
    sentence = " ".join(token for sentence in seg for token in sentence)
    return sentence.lower()


def transform(sentence):
    sentence = segmentating_sentence(sentence)
    encoded = tokenizer.encode_plus(sentence, add_special_tokens=True,
                                    max_length=MAX_LEN, padding='max_length',
                                    truncation=True,
                                    return_attention_mask=True)
    sequence = encoded.get("input_ids")
    attention_mask = encoded.get('attention_mask')

    return torch.Tensor(sequence), torch.Tensor(attention_mask)


class RatingDataset(Dataset):
    def __init__(self, sentences):
        super(RatingDataset, self).__init__()
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        return transform(sentence)


class SentimentAnalyzer(nn.Module):
    def __init__(self):
        super(SentimentAnalyzer, self).__init__()
        d_in, d_out = 768, 1
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.classifier = nn.Linear(d_in, d_out)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids,
                               attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, -1]
        logit = self.classifier(last_hidden_state_cls)

        return logit


model = SentimentAnalyzer()
model.load_state_dict(load("model.pt", map_location=device))
model.eval()


def predict(sentence):
    sequence, attention_mask = transform(sentence)
    with torch.no_grad():
        logits = model(torch.unsqueeze(sequence.long(), dim=0),
                       torch.unsqueeze(attention_mask.long(), dim=0))

    prob = logits.sigmoid().tolist()[0][0]
    return prob > THRESHOLD


def predict_many(sentences):
    all_logits = []

    dataset = RatingDataset(sentences)
    data_loader = DataLoader(dataset, 8)

    for batch in data_loader:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids.long(), b_attn_mask.long())
        all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    probs = all_logits.sigmoid().cpu().numpy()

    preds = np.zeros_like(probs)
    preds[:, 0] = np.where(probs[:, 0] > THRESHOLD, 1, 0)

    return probs, preds
