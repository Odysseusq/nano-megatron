import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_dataset(path, tokenizer, seq_len=2048):
    ds = load_dataset("json", data_files=path, split="train")

    def tokenize(example):
        tokens = tokenizer.encode(example["text"])[:seq_len]
        return {"tokens": tokens}

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["tokens"]) > 1)
    ds.set_format("numpy")
    return ds


def collate_packed(batch):
    input_ids, labels, positions, cu_seqlens = [], [], [], [0]
    max_seqlen = 0
    for sample in batch:
        tokens = sample["tokens"].tolist()
        n = len(tokens) - 1
        input_ids.extend(tokens[:-1])
        labels.extend(tokens[1:])
        positions.extend(range(n))
        cu_seqlens.append(cu_seqlens[-1] + n)
        max_seqlen = max(max_seqlen, n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device="cpu"),
        "labels": torch.tensor(labels, dtype=torch.long, device="cpu"),
        "positions": torch.tensor(positions, dtype=torch.long, device="cpu"),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32, device="cpu"),
        "max_seqlen": max_seqlen,
    }


def get_dataloader(path, tokenizer, batch_size=4, seq_len=2048, **kwargs):
    ds = get_dataset(path, tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_packed, **kwargs)
