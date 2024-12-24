"""Main module."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


class TokenizedDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx].squeeze(0) for k, v in self.inputs.items()}


def get_ngram_idx(input_ids, ngram_range=(3, 6)):
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy()
    ngram_idx = []
    for ngram_size in range(ngram_range[0], ngram_range[1] + 1):
        idx = np.vstack(
            [np.arange(input_ids.shape[1]) + i for i in range(ngram_size)]
        ).T
        idx = idx[(idx[:, -1] < input_ids.shape[1])]
        ngram_idx.append(idx)
    return ngram_idx


class Grammatron:
    def __init__(self, model_name, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(device)

    @property
    def device(self):
        return self.model.device

    def extract_ngrams(self, input_ids, token_embeds, ngram_range=(3, 6)):
        ngrams = []
        ngram_vecs = []
        ngram_idx = get_ngram_idx(input_ids, ngram_range=ngram_range)
        valid_token_mask = np.isin(
            input_ids, self.tokenizer.all_special_ids, invert=True
        )
        # Extract mean token embeddings for each ngram
        # (there is probably a faster way to do this without iterating over each sequence)
        for i in tqdm(
            range(input_ids.shape[0]), desc="Extracting", total=input_ids.shape[0]
        ):
            ngrams.append([])
            ngram_vecs.append([])
            for idx in ngram_idx:
                valid_ngrams = np.all(valid_token_mask[i, idx], axis=1)
                ngram_vecs[i].append(token_embeds[i, idx][valid_ngrams].mean(axis=1))
                ngrams[i].extend(
                    self.tokenizer.batch_decode(input_ids[i, idx][valid_ngrams])
                )
            ngram_vecs[i] = np.vstack(ngram_vecs[i])
        return ngrams, ngram_vecs

    def encode(
        self, docs, max_length=512, batch_size=32, amp=True, amp_dtype=torch.bfloat16
    ):
        if max_length is None:
            if self.tokenizer.model_max_length is None:
                raise ValueError(
                    "max_length must be specified if tokenizer.model_max_length is None"
                )
            max_length = self.tokenizer.model_max_length
            print(f"max_length set to {max_length}")
        inputs = self.tokenizer(
            docs,
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        loader = DataLoader(
            TokenizedDataset(inputs),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        token_embeds = []
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp, dtype=amp_dtype):
                for batch in tqdm(loader, desc="Encoding"):
                    batch = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()
                    }
                    token_embeds.append(
                        self.model(**batch).last_hidden_state.cpu().numpy()
                    )
        token_embeds = np.concatenate(token_embeds, axis=0)
        return inputs, token_embeds

    def encode_extract(
        self,
        docs,
        max_length=512,
        batch_size=32,
        ngram_range=(3, 6),
        amp=True,
        amp_dtype=torch.bfloat16,
    ):
        inputs, token_embeds = self.encode(docs, max_length, batch_size, amp, amp_dtype)
        ngrams, ngram_vecs = self.extract_ngrams(
            inputs["input_ids"], token_embeds, ngram_range=ngram_range
        )
        return ngrams, ngram_vecs

    def encode_queries(
        self, queries, max_length=512, batch_size=32, amp=True, amp_dtype=torch.bfloat16
    ):
        inputs, token_embeds = self.encode(
            queries,
            max_length=max_length,
            batch_size=batch_size,
            amp=amp,
            amp_dtype=amp_dtype,
        )
        valid_token_mask = np.isin(
            inputs["input_ids"], self.tokenizer.all_special_ids, invert=True
        )
        # Extract mean token embeddings for each query
        query_embeds = []
        for i in tqdm(
            range(inputs["input_ids"].shape[0]),
            desc="Extracting",
            total=inputs["input_ids"].shape[0],
        ):
            query_embeds.append(token_embeds[i, valid_token_mask[i], :].mean(axis=0))
        return np.vstack(query_embeds)
