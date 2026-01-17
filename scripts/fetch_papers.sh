#!/bin/bash
# Fetch research papers referenced in LITERATURE.md
# Papers are from arXiv and freely available

set -e

OUTDIR="References"
mkdir -p "$OUTDIR"

echo "Fetching research papers..."

# Late Chunking (Jina AI)
wget -q --show-progress -O "$OUTDIR/Late_Chunking_2409.04701.pdf" \
    "https://arxiv.org/pdf/2409.04701"

# SeDR: Segment Representation Learning
wget -q --show-progress -O "$OUTDIR/SeDR_2211.10841.pdf" \
    "https://arxiv.org/pdf/2211.10841"

# LongEmbed: Extending Embedding Models
wget -q --show-progress -O "$OUTDIR/LongEmbed_2404.12096.pdf" \
    "https://arxiv.org/pdf/2404.12096"

# Hierarchical Segmentation for RAG
wget -q --show-progress -O "$OUTDIR/Hierarchical_Segmentation_RAG_2507.09935.pdf" \
    "https://arxiv.org/pdf/2507.09935"

# ColBERT: Late Interaction over BERT
wget -q --show-progress -O "$OUTDIR/ColBERT_2004.12832.pdf" \
    "https://arxiv.org/pdf/2004.12832"

# ColBERTv2: Lightweight Late Interaction
wget -q --show-progress -O "$OUTDIR/ColBERTv2_2112.01488.pdf" \
    "https://arxiv.org/pdf/2112.01488"

echo ""
echo "Done. Papers saved to $OUTDIR/"
ls -lh "$OUTDIR"
