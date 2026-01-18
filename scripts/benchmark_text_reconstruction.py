"""Benchmark text reconstruction vs decoding approaches.

Compares:
- _decode_chunks: Original parallelized tokenizer.batch_decode
- _reconstruct_chunk_texts: New single-threaded sentence joining
- Optimized variants with min/max instead of torch.unique
- Parallel reconstruction with different batch sizes
"""

import time
from functools import partial

import polars as pl
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

from afterthoughts import Encoder
from afterthoughts.tokenize import DynamicTokenSampler, dynamic_pad_collate


def reconstruct_optimized_single(
    chunk_sentence_ids, chunk_token_ids, sequence_idx, seq_to_doc, sentence_texts
):
    """Optimized single-threaded reconstruction using min/max instead of unique."""
    chunks = []
    flat_sent_ids = [sid for batch in chunk_sentence_ids for sid in batch]

    for i, sent_ids in enumerate(flat_sent_ids):
        doc_idx = seq_to_doc[sequence_idx[i].item()]
        doc_sentences = sentence_texts[doc_idx]
        mask = sent_ids != -1
        if mask.any():
            valid = sent_ids[mask]
            min_id, max_id = valid[0].item(), valid[-1].item()
            if max_id < len(doc_sentences):
                chunks.append(" ".join(doc_sentences[sid] for sid in range(min_id, max_id + 1)))
            else:
                chunks.append(None)  # Fallback marker
        else:
            chunks.append("")
    return pl.Series(chunks)


def reconstruct_chunk_batch(batch_sent_ids, start_idx, sequence_idx, seq_to_doc, sentence_texts):
    """Process a batch of chunks for parallel execution."""
    chunks = []
    for i, sent_ids in enumerate(batch_sent_ids):
        doc_idx = seq_to_doc[sequence_idx[start_idx + i].item()]
        doc_sentences = sentence_texts[doc_idx]
        mask = sent_ids != -1
        if mask.any():
            valid = sent_ids[mask]
            min_id, max_id = valid[0].item(), valid[-1].item()
            if max_id < len(doc_sentences):
                chunks.append(" ".join(doc_sentences[sid] for sid in range(min_id, max_id + 1)))
            else:
                chunks.append(None)
        else:
            chunks.append("")
    return chunks


def reconstruct_parallel_batched(
    chunk_sentence_ids,
    sequence_idx,
    seq_to_doc,
    sentence_texts,
    n_jobs=-1,
    batch_size=1000,
):
    """Parallel reconstruction with configurable batch size.

    Instead of parallelizing per-model-batch, we flatten all chunks and
    re-batch into larger chunks of work to reduce joblib overhead.
    """
    # Flatten all sentence IDs
    flat_sent_ids = [sid for batch in chunk_sentence_ids for sid in batch]
    n_chunks = len(flat_sent_ids)

    # Create batches of specified size
    batches = []
    start_indices = []
    for i in range(0, n_chunks, batch_size):
        batches.append(flat_sent_ids[i : i + batch_size])
        start_indices.append(i)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(reconstruct_chunk_batch)(batch, start_idx, sequence_idx, seq_to_doc, sentence_texts)
        for batch, start_idx in zip(batches, start_indices, strict=False)
    )
    return pl.Series([c for batch in results for c in batch])


def prepare_data(model, docs):
    """Tokenize and generate chunk data for benchmarking."""
    inputs, sentence_texts = model._tokenize(
        docs, max_length=128, prechunk=True, show_progress=False
    )
    loader = DataLoader(
        inputs,
        shuffle=False,
        pin_memory=False,
        batch_sampler=DynamicTokenSampler(inputs, max_tokens=512),
        collate_fn=partial(dynamic_pad_collate, pad_token_id=model.tokenizer.pad_token_id),
    )

    chunk_token_ids_batched = []
    chunk_sentence_ids_batched = []
    sequence_idx_list = []

    for batch in model._generate_chunk_embeds(
        loader, num_sents=1, chunk_overlap=0, show_progress=False
    ):
        chunk_token_ids_batched.append(batch["chunk_token_ids"])
        chunk_sentence_ids_batched.append(batch["sentence_ids"])
        sequence_idx_list.append(batch["sequence_idx"])

    sequence_idx = torch.cat(sequence_idx_list)
    seq_to_doc = dict(
        zip(
            inputs.data["sequence_idx"],
            inputs.data["overflow_to_sample_mapping"],
            strict=False,
        )
    )
    n_chunks = sum(t.shape[0] for t in chunk_token_ids_batched)

    return {
        "chunk_token_ids": chunk_token_ids_batched,
        "chunk_sentence_ids": chunk_sentence_ids_batched,
        "sequence_idx": sequence_idx,
        "seq_to_doc": seq_to_doc,
        "sentence_texts": sentence_texts,
        "n_chunks": n_chunks,
    }


def benchmark(fn, num_runs=3):
    """Run a benchmark and return average time in ms."""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times) * 1000
    return avg


def run_benchmarks(n_docs_list, sents_per_doc=20):
    """Run full benchmark suite."""
    model = Encoder(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        device="cpu",
        _num_token_jobs=-1,
    )

    print("=" * 120)
    print("Text Reconstruction Benchmark")
    print("=" * 120)
    print()

    for n_docs in n_docs_list:
        docs = [" ".join([f"Sentence number {i}." for i in range(sents_per_doc)])] * n_docs

        data = prepare_data(model, docs)
        n_chunks = data["n_chunks"]

        print(f"Dataset: {n_docs} docs, {n_chunks} chunks")
        print("-" * 80)

        # Benchmark decode (parallelized)
        t = benchmark(
            lambda d=data: model._decode_chunks(d["chunk_token_ids"], show_progress=False),
        )
        print(f"  _decode_chunks (parallel):        {t:>8.1f}ms")

        # Benchmark original reconstruct
        t = benchmark(
            lambda d=data: model._reconstruct_chunk_texts(
                d["chunk_sentence_ids"],
                d["chunk_token_ids"],
                d["sequence_idx"],
                d["seq_to_doc"],
                d["sentence_texts"],
                show_progress=False,
            ),
        )
        print(f"  _reconstruct (original):          {t:>8.1f}ms")

        # Benchmark optimized single-threaded
        t = benchmark(
            lambda d=data: reconstruct_optimized_single(
                d["chunk_sentence_ids"],
                d["chunk_token_ids"],
                d["sequence_idx"],
                d["seq_to_doc"],
                d["sentence_texts"],
            ),
        )
        print(f"  reconstruct (opt single):         {t:>8.1f}ms")

        # Benchmark parallel with different batch sizes
        for batch_size in [100, 500, 1000, 5000]:
            if batch_size > n_chunks:
                continue
            t = benchmark(
                lambda bs=batch_size, d=data: reconstruct_parallel_batched(
                    d["chunk_sentence_ids"],
                    d["sequence_idx"],
                    d["seq_to_doc"],
                    d["sentence_texts"],
                    n_jobs=-1,
                    batch_size=bs,
                ),
            )
            print(f"  reconstruct (parallel, b={batch_size:<5}): {t:>8.1f}ms")

        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark text reconstruction methods")
    parser.add_argument(
        "--docs",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000, 5000],
        help="Number of documents to test",
    )
    parser.add_argument(
        "--sents",
        type=int,
        default=20,
        help="Sentences per document",
    )
    args = parser.parse_args()

    run_benchmarks(args.docs, args.sents)
