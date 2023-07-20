import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class Tabdataset(Dataset):
    def __init__(self, data, label, start, end):
        self.length = len(data)
        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(label)
        self.start = start
        self.end = end

    def __getitem__(self, item):
        return self.data[item][self.start:self.end], self.label[item]

    def __len__(self):
        return self.length


class Textdataset(Dataset):
    def __init__(self, data, label, start, end, max_len=160):
        self.length = len(data)
        self.data = data
        self.label = torch.LongTensor(label)
        self.start = start
        self.end = end
        self.max_len = max_len
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

    def __getitem__(self, item):
        token_ids = self.tokenizer.encode(text=self.data[item][self.start:self.end], add_special_tokens=True,
                                          max_length=self.max_len, padding='max_length', truncation=True,
                                          return_tensors='pt').view(-1)
        return token_ids, self.label[item]

    def __len__(self):
        return self.length
