import torch
from torch import nn, load
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar",
                         annotators="wseg", max_heap_size='-Xmx500m')

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)


MAX_LEN = 150


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
    print(type(sentence))
    with torch.no_grad():
        logits = model(sequence, attention_mask)

    print(logits)
