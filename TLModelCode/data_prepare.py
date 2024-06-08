import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


def text_pipeline(string, vocab):
    return vocab(tokenizer(string))


def clean_data(df):
    df_cleaned = df.iloc[:, 0:2]
    df_cleaned = df_cleaned.iloc[:1000000, :]
    df_cleaned["url"] = df_cleaned["url"].transform(
        lambda x: x.replace(".", " . ").replace("-", " - ").replace("/", " / "))
    return df_cleaned


class CustomTorchDataset(Dataset):
    def __init__(self, df):
        self.df_cleaned = clean_data(df)

        urls = self.df_cleaned["url"].values
        self.vocab = build_vocab_from_iterator(yield_tokens(urls), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

        self.df_cleaned["text_seq"] = [torch.tensor(text_pipeline(url, self.vocab)) for url in urls]
        self.ps = torch.nn.utils.rnn.pad_sequence(self.df_cleaned["text_seq"].values, batch_first=True, padding_value=0.0)

    def __len__(self):
        return len(self.df_cleaned)

    def __getitem__(self, idx):
        url_seq_tens = self.ps[idx]
        label = self.df_cleaned.iloc[idx, 1]
        return url_seq_tens, label
