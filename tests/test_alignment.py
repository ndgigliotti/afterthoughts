"""
Tests for embedding-segment alignment using semantic search.

These tests verify that embeddings are correctly aligned with their corresponding
text segments after all the sorting and reordering that happens during encoding.
This catches bugs like sequence index mapping errors that cause embeddings to be
swapped between documents.

The key bug scenario this catches:
- Documents are sorted by token count for efficient batching
- If doc 0 is shorter than doc 1, after sorting: batch[0]=doc1, batch[1]=doc0
- token_embeds[0] contains doc1's embeddings, token_embeds[1] contains doc0's
- A bug in mapping sequence_idx to batch positions can swap embeddings between docs

Note: These tests require downloading a model and are slower than unit tests.
"""

import numpy as np
from numpy.linalg import norm

from afterthoughts import Encoder


def encode_text_as_query(model: Encoder, text: str) -> np.ndarray:
    """Encode text using the query encoder for comparison."""
    return model.encode_queries([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def get_avg_similarity(
    model: Encoder,
    query: str,
    embeddings: np.ndarray,
    doc_indices: list[int],
    target_doc: int,
) -> tuple[float, float]:
    """
    Compute average similarity for target doc vs other docs.

    Returns
    -------
    tuple[float, float]
        (avg_similarity_target_doc, avg_similarity_other_docs)
    """
    q = model.encode_queries([query])[0]

    target_sims = []
    other_sims = []

    for i, doc_idx in enumerate(doc_indices):
        sim = cosine_similarity(q, embeddings[i])
        if doc_idx == target_doc:
            target_sims.append(sim)
        else:
            other_sims.append(sim)

    avg_target = np.mean(target_sims) if target_sims else 0.0
    avg_other = np.mean(other_sims) if other_sims else 0.0

    return avg_target, avg_other


class TestEmbeddingAlignment:
    """Tests that embeddings are correctly aligned with their text segments."""

    def test_two_documents_alignment(self, model):
        """Test alignment with two semantically distinct documents."""
        docs = [
            "Machine learning is AI. It enables learning.",
            "Python is a programming language. It is popular.",
        ]

        df, X = model.encode(docs, max_chunk_sents=[1, 2], chunk_overlap=0, show_progress=False)

        # ML segments should have higher similarity to "machine learning" than Python segments
        ml_avg, py_avg = get_avg_similarity(
            model,
            query="machine learning artificial intelligence",
            embeddings=X,
            doc_indices=df["document_idx"].to_list(),
            target_doc=0,
        )

        assert ml_avg > py_avg, (
            f"ML segments should have higher similarity to ML query. "
            f"Got ML avg={ml_avg:.3f}, Python avg={py_avg:.3f}"
        )
        # Expect a meaningful difference, not just noise
        assert ml_avg - py_avg > 0.2, (
            f"Similarity difference too small. "
            f"ML avg={ml_avg:.3f}, Python avg={py_avg:.3f}, diff={ml_avg - py_avg:.3f}"
        )

    def test_two_documents_reverse_query(self, model):
        """Test alignment with query matching the second document."""
        docs = [
            "Machine learning is AI. It enables learning.",
            "Python is a programming language. It is popular.",
        ]

        df, X = model.encode(docs, max_chunk_sents=[1, 2], chunk_overlap=0, show_progress=False)

        # Python segments should have higher similarity to "python programming"
        py_avg, ml_avg = get_avg_similarity(
            model,
            query="python programming language",
            embeddings=X,
            doc_indices=df["document_idx"].to_list(),
            target_doc=1,
        )

        assert py_avg > ml_avg, (
            f"Python segments should have higher similarity to Python query. "
            f"Got Python avg={py_avg:.3f}, ML avg={ml_avg:.3f}"
        )
        assert py_avg - ml_avg > 0.2, (
            f"Similarity difference too small. "
            f"Python avg={py_avg:.3f}, ML avg={ml_avg:.3f}, diff={py_avg - ml_avg:.3f}"
        )

    def test_five_documents_alignment(self, model):
        """Test alignment with multiple semantically distinct documents."""
        docs = [
            "Machine learning is AI. It enables learning.",
            "Python is a programming language. It is popular.",
            "The quick brown fox jumps. Over the lazy dog.",
            "Hello world is a classic. It is used for testing.",
            "Deep learning uses neural networks. They learn representations.",
        ]

        df, X = model.encode(docs, max_chunk_sents=[1, 2], chunk_overlap=0, show_progress=False)
        doc_indices = df["document_idx"].to_list()

        # Test each document with a relevant query
        test_cases = [
            (0, "machine learning artificial intelligence"),
            (1, "python programming language"),
            (2, "quick brown fox lazy dog"),
            (3, "hello world testing"),
            (4, "deep learning neural networks"),
        ]

        for target_doc, query in test_cases:
            target_avg, other_avg = get_avg_similarity(
                model,
                query=query,
                embeddings=X,
                doc_indices=doc_indices,
                target_doc=target_doc,
            )

            assert target_avg > other_avg, (
                f"Doc {target_doc} segments should have higher similarity to '{query}'. "
                f"Got target avg={target_avg:.3f}, other avg={other_avg:.3f}"
            )

    def test_alignment_with_single_segment_size(self, model):
        """Test alignment with only single-sentence segments."""
        docs = [
            "Machine learning is AI. It enables learning.",
            "Python is a programming language. It is popular.",
        ]

        df, X = model.encode(docs, max_chunk_sents=[1], chunk_overlap=0, show_progress=False)

        ml_avg, py_avg = get_avg_similarity(
            model,
            query="machine learning",
            embeddings=X,
            doc_indices=df["document_idx"].to_list(),
            target_doc=0,
        )

        assert (
            ml_avg > py_avg
        ), f"Single segment size: ML avg={ml_avg:.3f} should be > Python avg={py_avg:.3f}"

    def test_alignment_with_larger_max_chunk_sents(self, model):
        """Test alignment with larger segment sizes."""
        docs = [
            "Machine learning is AI. It enables automatic learning. Models improve over time. Data is key.",
            "Python is a programming language. It is popular for scripting. Easy to learn. Great community.",
        ]

        df, X = model.encode(docs, max_chunk_sents=[2, 3], chunk_overlap=0, show_progress=False)

        ml_avg, py_avg = get_avg_similarity(
            model,
            query="machine learning AI models",
            embeddings=X,
            doc_indices=df["document_idx"].to_list(),
            target_doc=0,
        )

        assert (
            ml_avg > py_avg
        ), f"Larger segments: ML avg={ml_avg:.3f} should be > Python avg={py_avg:.3f}"

    def test_alignment_with_overlap(self, model):
        """Test alignment with overlapping segments."""
        docs = [
            "Machine learning is AI. It enables learning. Models improve.",
            "Python is a programming language. It is popular. Easy to use.",
        ]

        df, X = model.encode(docs, max_chunk_sents=[2], chunk_overlap=1, show_progress=False)

        ml_avg, py_avg = get_avg_similarity(
            model,
            query="machine learning",
            embeddings=X,
            doc_indices=df["document_idx"].to_list(),
            target_doc=0,
        )

        assert (
            ml_avg > py_avg
        ), f"With overlap: ML avg={ml_avg:.3f} should be > Python avg={py_avg:.3f}"

    def test_alignment_preserves_document_order(self, model):
        """Test that document_idx correctly identifies source documents."""
        docs = [
            "Cats are furry pets. They purr when happy.",
            "Dogs are loyal companions. They bark to communicate.",
            "Birds can fly. They have feathers and wings.",
        ]

        df, X = model.encode(docs, max_chunk_sents=[1, 2], chunk_overlap=0, show_progress=False)

        # Query for cats should match doc 0
        q_cats = model.encode_queries(["cats furry pets purr"])[0]
        cat_sims = [
            (i, cosine_similarity(q_cats, X[i]), df["document_idx"][i]) for i in range(len(X))
        ]
        top_cat = max(cat_sims, key=lambda x: x[1])
        assert top_cat[2] == 0, f"Top result for 'cats' should be doc 0, got doc {top_cat[2]}"

        # Query for dogs should match doc 1
        q_dogs = model.encode_queries(["dogs loyal bark"])[0]
        dog_sims = [
            (i, cosine_similarity(q_dogs, X[i]), df["document_idx"][i]) for i in range(len(X))
        ]
        top_dog = max(dog_sims, key=lambda x: x[1])
        assert top_dog[2] == 1, f"Top result for 'dogs' should be doc 1, got doc {top_dog[2]}"

        # Query for birds should match doc 2
        q_birds = model.encode_queries(["birds fly feathers wings"])[0]
        bird_sims = [
            (i, cosine_similarity(q_birds, X[i]), df["document_idx"][i]) for i in range(len(X))
        ]
        top_bird = max(bird_sims, key=lambda x: x[1])
        assert top_bird[2] == 2, f"Top result for 'birds' should be doc 2, got doc {top_bird[2]}"

    def test_alignment_with_varying_document_lengths(self, model):
        """Test alignment when documents have very different lengths."""
        docs = [
            "Short ML doc.",  # Very short
            "Python is a programming language. It is popular. Easy to learn. Great for beginners. Has many libraries. Used in data science. Also for web development. Very versatile.",  # Long
            "Medium length about cats. Cats are pets.",  # Medium
        ]

        df, X = model.encode(docs, max_chunk_sents=[1, 2], chunk_overlap=0, show_progress=False)

        # Python query should match doc 1
        py_avg, other_avg = get_avg_similarity(
            model,
            query="python programming language libraries",
            embeddings=X,
            doc_indices=df["document_idx"].to_list(),
            target_doc=1,
        )

        assert (
            py_avg > other_avg
        ), f"Varying lengths: Python avg={py_avg:.3f} should be > other avg={other_avg:.3f}"

    def test_segment_text_matches_embedding(self, model):
        """Test that each segment's text semantically matches its embedding."""
        docs = [
            "Machine learning enables AI.",
            "Python is for programming.",
        ]

        df, X = model.encode(docs, max_chunk_sents=[1], chunk_overlap=0, show_progress=False)

        # For each segment, verify its embedding matches its text better than other segments
        for i in range(len(df)):
            segment_text = df["chunk"][i]
            segment_embed = X[i]

            # Encode the segment text as a query
            q = model.encode_queries([segment_text])[0]

            # This segment's embedding should have highest similarity to its own text
            self_sim = cosine_similarity(q, segment_embed)

            other_sims = [cosine_similarity(q, X[j]) for j in range(len(X)) if j != i]

            # Self-similarity should be highest (or very close to highest)
            max_other = max(other_sims) if other_sims else 0
            assert self_sim >= max_other - 0.1, (
                f"Segment {i} '{segment_text[:30]}...' self-sim={self_sim:.3f} "
                f"should be >= max_other={max_other:.3f}"
            )


class TestAlignmentWithChunking:
    """Tests for alignment when documents are chunked due to length limits."""

    def test_chunked_document_alignment(self, model):
        """Test alignment when a document is split into chunks."""
        # Create a document long enough to potentially be chunked
        long_ml_doc = " ".join(
            [
                "Machine learning is a subset of artificial intelligence.",
                "It enables computers to learn from data.",
                "Neural networks are inspired by the brain.",
                "Deep learning uses multiple layers.",
                "Training requires large datasets.",
                "Models can recognize patterns.",
                "Applications include image recognition.",
                "Natural language processing uses ML.",
            ]
        )

        short_py_doc = "Python is a programming language. It is popular."

        docs = [long_ml_doc, short_py_doc]

        df, X = model.encode(
            docs,
            max_chunk_sents=[1, 2],
            chunk_overlap=0,
            max_length=128,  # Force chunking
            prechunk=True,
            show_progress=False,
        )

        # All ML segments (doc 0) should have higher similarity to ML query
        ml_avg, py_avg = get_avg_similarity(
            model,
            query="machine learning neural networks deep learning",
            embeddings=X,
            doc_indices=df["document_idx"].to_list(),
            target_doc=0,
        )

        assert (
            ml_avg > py_avg
        ), f"Chunked doc: ML avg={ml_avg:.3f} should be > Python avg={py_avg:.3f}"

    def test_multiple_chunks_same_document(self, model):
        """Test that multiple chunks from the same document are all aligned."""
        # Create document that will definitely be chunked
        sentences = [
            f"Machine learning concept number {i}. This is about AI and data science."
            for i in range(20)
        ]
        long_doc = " ".join(sentences)

        docs = [long_doc, "Python programming is fun. Code is easy."]

        df, X = model.encode(
            docs,
            max_chunk_sents=[1],
            chunk_overlap=0,
            max_length=64,  # Force aggressive chunking
            prechunk=True,
            show_progress=False,
        )

        # Count segments per document
        doc_0_count = sum(1 for d in df["document_idx"] if d == 0)
        doc_1_count = sum(1 for d in df["document_idx"] if d == 1)

        # Doc 0 should have more segments due to chunking
        assert (
            doc_0_count > doc_1_count
        ), f"Long doc should have more segments: doc_0={doc_0_count}, doc_1={doc_1_count}"

        # All doc 0 segments should still align with ML query
        ml_avg, py_avg = get_avg_similarity(
            model,
            query="machine learning AI data science",
            embeddings=X,
            doc_indices=df["document_idx"].to_list(),
            target_doc=0,
        )

        assert (
            ml_avg > py_avg
        ), f"All chunks should align: ML avg={ml_avg:.3f} > Python avg={py_avg:.3f}"


class TestAlignmentEdgeCases:
    """Tests for edge cases in embedding alignment."""

    def test_single_document(self, model):
        """Test alignment with a single document."""
        docs = ["Machine learning is AI. It enables learning."]

        df, X = model.encode(docs, max_chunk_sents=[1, 2], chunk_overlap=0, show_progress=False)

        # All segments should come from doc 0
        assert all(d == 0 for d in df["document_idx"]), "All segments should be from doc 0"

        # Segments should have reasonable similarity to relevant query
        q = model.encode_queries(["machine learning"])[0]
        sims = [cosine_similarity(q, X[i]) for i in range(len(X))]
        avg_sim = np.mean(sims)

        assert avg_sim > 0.3, f"Single doc segments should have decent similarity: {avg_sim:.3f}"

    def test_identical_documents(self, model):
        """Test alignment when documents are identical."""
        doc_text = "Machine learning is AI. It enables learning."
        docs = [doc_text, doc_text]

        df, X = model.encode(docs, max_chunk_sents=[1], chunk_overlap=0, show_progress=False)

        # Get segments from each document, grouped by segment_idx
        doc_0_indices = [i for i, d in enumerate(df["document_idx"]) if d == 0]
        doc_1_indices = [i for i, d in enumerate(df["document_idx"]) if d == 1]

        # Same number of segments per document
        assert (
            len(doc_0_indices) == len(doc_1_indices)
        ), f"Identical docs should have same segment count: {len(doc_0_indices)} vs {len(doc_1_indices)}"

        # Corresponding segments (by segment_idx) should have nearly identical embeddings
        for i, (idx_0, idx_1) in enumerate(zip(doc_0_indices, doc_1_indices, strict=False)):
            sim = cosine_similarity(X[idx_0], X[idx_1])
            assert (
                sim > 0.99
            ), f"Segment {i}: identical docs should have near-identical embeddings: {sim:.3f}"

    def test_many_short_documents(self, model):
        """Test alignment with many short documents."""
        topics = [
            ("cats", "Cats are pets."),
            ("dogs", "Dogs are loyal."),
            ("birds", "Birds can fly."),
            ("fish", "Fish swim in water."),
            ("horses", "Horses can run fast."),
            ("elephants", "Elephants are large."),
            ("lions", "Lions are predators."),
            ("pandas", "Pandas eat bamboo."),
        ]

        docs = [text for _, text in topics]
        df, X = model.encode(docs, max_chunk_sents=[1], chunk_overlap=0, show_progress=False)

        # Each topic query should match its corresponding document
        for doc_idx, (topic, _) in enumerate(topics):
            q = model.encode_queries([topic])[0]
            sims = [(i, cosine_similarity(q, X[i]), df["document_idx"][i]) for i in range(len(X))]
            top_result = max(sims, key=lambda x: x[1])

            assert (
                top_result[2] == doc_idx
            ), f"Query '{topic}' should match doc {doc_idx}, got doc {top_result[2]}"

    def test_debug_mode_alignment(self, model):
        """Test that debug mode doesn't affect alignment."""
        docs = [
            "Machine learning is AI. It enables learning.",
            "Python is a programming language. It is popular.",
        ]

        # Encode with and without debug mode
        df_normal, X_normal = model.encode(
            docs, max_chunk_sents=[1, 2], chunk_overlap=0, debug=False, show_progress=False
        )
        df_debug, X_debug = model.encode(
            docs, max_chunk_sents=[1, 2], chunk_overlap=0, debug=True, show_progress=False
        )

        # Embeddings should be identical
        np.testing.assert_array_almost_equal(
            X_normal,
            X_debug,
            decimal=5,
            err_msg="Debug mode should not affect embeddings",
        )

        # Chunks should be identical
        assert (
            df_normal["chunk"].to_list() == df_debug["chunk"].to_list()
        ), "Debug mode should not affect chunk order"


class TestBatchReorderingAlignment:
    """
    Tests specifically targeting the batch reordering bug.

    The bug: When sequences are sorted by length for efficient batching,
    the mapping from sequence_idx to batch position must use the actual
    batch order, not sorted numerical order. Using torch.unique() which
    sorts values numerically caused embeddings to be swapped between documents.

    These tests create scenarios where batch reordering MUST occur
    and verify alignment is preserved.
    """

    def test_short_doc_first_triggers_reorder(self, model):
        """
        Critical test: Short doc at index 0, long doc at index 1.

        After sorting by length:
        - batch position 0 = doc 1 (long, more tokens)
        - batch position 1 = doc 0 (short, fewer tokens)

        This is the exact scenario that triggered the original bug.
        """
        # Short document (fewer tokens) at index 0
        short_doc = "Cats meow loudly."
        # Long document (more tokens) at index 1
        long_doc = "Dogs bark and play. Dogs are loyal companions. Dogs love to fetch balls."

        docs = [short_doc, long_doc]
        df, X = model.encode(docs, max_chunk_sents=[1], chunk_overlap=0, show_progress=False)

        # Get average embedding per document
        doc_0_embeds = np.array([X[i] for i in range(len(df)) if df["document_idx"][i] == 0])
        doc_1_embeds = np.array([X[i] for i in range(len(df)) if df["document_idx"][i] == 1])

        doc_0_avg = doc_0_embeds.mean(axis=0)
        doc_1_avg = doc_1_embeds.mean(axis=0)

        # Doc 0 (cats) should match cat query better than dog query
        cat_query = encode_text_as_query(model, "cats meow feline")
        dog_query = encode_text_as_query(model, "dogs bark canine")

        doc_0_cat_sim = cosine_similarity(cat_query, doc_0_avg)
        doc_0_dog_sim = cosine_similarity(dog_query, doc_0_avg)
        assert (
            doc_0_cat_sim > doc_0_dog_sim
        ), f"Doc 0 (cats) embedding misaligned: cat_sim={doc_0_cat_sim:.3f} should be > dog_sim={doc_0_dog_sim:.3f}"

        # Doc 1 (dogs) should match dog query better than cat query
        doc_1_cat_sim = cosine_similarity(cat_query, doc_1_avg)
        doc_1_dog_sim = cosine_similarity(dog_query, doc_1_avg)
        assert (
            doc_1_dog_sim > doc_1_cat_sim
        ), f"Doc 1 (dogs) embedding misaligned: dog_sim={doc_1_dog_sim:.3f} should be > cat_sim={doc_1_cat_sim:.3f}"

    def test_long_doc_first_no_reorder(self, model):
        """
        Control test: Long doc at index 0, short doc at index 1.

        After sorting by length, order remains the same.
        This should work even with a buggy implementation.
        """
        # Long document at index 0
        long_doc = "Dogs bark and play. Dogs are loyal companions. Dogs love to fetch balls."
        # Short document at index 1
        short_doc = "Cats meow loudly."

        docs = [long_doc, short_doc]
        df, X = model.encode(docs, max_chunk_sents=[1], chunk_overlap=0, show_progress=False)

        # Get average embedding per document
        doc_0_embeds = np.array([X[i] for i in range(len(df)) if df["document_idx"][i] == 0])
        doc_1_embeds = np.array([X[i] for i in range(len(df)) if df["document_idx"][i] == 1])

        doc_0_avg = doc_0_embeds.mean(axis=0)
        doc_1_avg = doc_1_embeds.mean(axis=0)

        cat_query = encode_text_as_query(model, "cats meow feline")
        dog_query = encode_text_as_query(model, "dogs bark canine")

        # Doc 0 (dogs) should match dog query better
        doc_0_dog_sim = cosine_similarity(dog_query, doc_0_avg)
        doc_0_cat_sim = cosine_similarity(cat_query, doc_0_avg)
        assert (
            doc_0_dog_sim > doc_0_cat_sim
        ), f"Doc 0 (dogs) misaligned: dog_sim={doc_0_dog_sim:.3f} should be > cat_sim={doc_0_cat_sim:.3f}"

        # Doc 1 (cats) should match cat query better
        doc_1_cat_sim = cosine_similarity(cat_query, doc_1_avg)
        doc_1_dog_sim = cosine_similarity(dog_query, doc_1_avg)
        assert (
            doc_1_cat_sim > doc_1_dog_sim
        ), f"Doc 1 (cats) misaligned: cat_sim={doc_1_cat_sim:.3f} should be > dog_sim={doc_1_dog_sim:.3f}"

    def test_multiple_reorderings(self, model):
        """
        Test with multiple documents that will be reordered.

        Input order by index: 0, 1, 2, 3
        Order by length (ascending): varies based on content
        """
        docs = [
            "A.",  # Shortest - index 0
            "Dogs are loyal companions who love their owners. They bark and play.",  # Long - index 1
            "Cats meow softly.",  # Medium - index 2
            "Birds fly high in the sky. They have colorful feathers and sing songs.",  # Longest - index 3
        ]

        df, _X = model.encode(docs, max_chunk_sents=[1], chunk_overlap=0, show_progress=False)

        # Create mapping of expected content per document
        expected_content = {
            0: ["a"],  # Just "A."
            1: ["dog", "loyal", "bark", "play"],
            2: ["cat", "meow"],
            3: ["bird", "fly", "feather", "sing"],
        }

        # Verify each document's segments contain expected content
        for i in range(len(df)):
            doc_idx = df["document_idx"][i]
            segment = df["chunk"][i].lower()

            # At least one expected keyword should be in segment
            keywords = expected_content[doc_idx]
            found = any(kw in segment for kw in keywords)
            assert found, f"Doc {doc_idx} segment '{segment}' should contain one of {keywords}"

    def test_embedding_text_direct_correspondence(self, model):
        """
        Rigorous test: Verify embedding[i] was computed from segment[i]'s content.

        For each row, encode the segment text as a query and verify it has
        the highest similarity to its own embedding (not just "high" similarity).
        """
        docs = [
            "Quantum physics studies atoms.",  # Very distinct topic
            "Medieval castles had moats.",  # Very distinct topic
        ]

        df, X = model.encode(docs, max_chunk_sents=[1], chunk_overlap=0, show_progress=False)

        # For each segment, its text encoded as query should match its embedding best
        for i in range(len(df)):
            segment_text = df["chunk"][i]

            # Encode this segment's text as a query
            query_embed = encode_text_as_query(model, segment_text)

            # Compute similarity to all embeddings
            similarities = [cosine_similarity(query_embed, X[j]) for j in range(len(X))]

            # This segment's embedding should have highest similarity
            max_sim_idx = np.argmax(similarities)
            assert max_sim_idx == i, (
                f"Segment {i} '{segment_text[:30]}' embedding mismatch: "
                f"highest similarity at index {max_sim_idx}, expected {i}. "
                f"Similarities: {[f'{s:.3f}' for s in similarities]}"
            )

    def test_three_docs_all_orderings(self, model):
        """
        Test all possible reorderings with 3 documents of distinct lengths.
        """
        # Create 3 very distinct documents with very different lengths
        topics = [
            ("quantum", "Quantum mechanics governs subatomic particles."),
            (
                "medieval",
                "Medieval knights wore heavy armor into battle. Castles provided defense.",
            ),
            (
                "space",
                "Astronauts explore outer space. Rockets launch from Earth. The moon has no atmosphere. Mars is called the red planet.",
            ),
        ]

        # Test different orderings
        orderings = [
            [0, 1, 2],  # short, medium, long
            [0, 2, 1],  # short, long, medium
            [1, 0, 2],  # medium, short, long
            [1, 2, 0],  # medium, long, short
            [2, 0, 1],  # long, short, medium
            [2, 1, 0],  # long, medium, short
        ]

        for order in orderings:
            docs = [topics[i][1] for i in order]
            topic_names = [topics[i][0] for i in order]

            df, X = model.encode(docs, max_chunk_sents=[1], chunk_overlap=0, show_progress=False)

            # Verify alignment for each document
            for doc_idx, topic_name in enumerate(topic_names):
                # Get embeddings for this document
                doc_embeds = [X[i] for i in range(len(df)) if df["document_idx"][i] == doc_idx]

                # All embeddings for this doc should match its topic query best
                for embed in doc_embeds:
                    sims = {
                        "quantum": cosine_similarity(
                            encode_text_as_query(model, "quantum mechanics particles"),
                            embed,
                        ),
                        "medieval": cosine_similarity(
                            encode_text_as_query(model, "medieval knights castles armor"),
                            embed,
                        ),
                        "space": cosine_similarity(
                            encode_text_as_query(model, "astronauts space rockets moon mars"),
                            embed,
                        ),
                    }

                    best_topic = max(sims, key=sims.get)
                    assert best_topic == topic_name, (
                        f"Order {order}: Doc {doc_idx} (topic={topic_name}) embedding "
                        f"matched {best_topic} instead. Sims: {sims}"
                    )

    def test_many_docs_stress_test(self, model):
        """
        Stress test with many documents of varying lengths.

        This tests that alignment is maintained even with many reorderings.
        """
        # Create documents with very distinct, identifiable content
        # Each document has unique keywords for semantic matching
        docs_data = [
            (
                "python indentation scripting",
                "Python programming uses indentation for code blocks.",
            ),
            (
                "java classes types",
                "Java requires explicit type declarations. Classes are fundamental in Java programming.",
            ),
            (
                "rust memory borrow",
                "Rust prevents memory errors. The borrow checker ensures safety. Rust is fast and safe.",
            ),
            (
                "go goroutines channels",
                "Go has goroutines for concurrency. Channels enable communication. Go compiles quickly.",
            ),
            (
                "javascript browser dom",
                "JavaScript runs in browsers and manipulates the DOM.",
            ),
            (
                "typescript interfaces static",
                "TypeScript adds static types to JavaScript. Interfaces define contracts and shapes.",
            ),
            (
                "ruby rails gems",
                "Ruby emphasizes developer happiness. Rails is a popular web framework. Gems are Ruby packages.",
            ),
            (
                "c pointers lowlevel",
                "C is a low-level language with pointers and manual memory management.",
            ),
        ]

        # Shuffle to create reordering
        import random

        random.seed(42)  # Reproducible
        shuffled = docs_data.copy()
        random.shuffle(shuffled)

        docs = [d[1] for d in shuffled]
        queries = [d[0] for d in shuffled]  # Use as queries

        df, X = model.encode(docs, max_chunk_sents=[1], chunk_overlap=0, show_progress=False)

        # For each document, its corresponding query should match it best
        for doc_idx, query in enumerate(queries):
            query_embed = encode_text_as_query(model, query)

            # Compute average similarity per document
            doc_sims = {}
            for d_idx in range(len(docs)):
                doc_embeds = [X[i] for i in range(len(df)) if df["document_idx"][i] == d_idx]
                if doc_embeds:
                    avg_embed = np.mean(doc_embeds, axis=0)
                    doc_sims[d_idx] = cosine_similarity(query_embed, avg_embed)

            # The target document should have highest similarity
            best_doc = max(doc_sims, key=doc_sims.get)
            assert best_doc == doc_idx, (
                f"Query '{query}' should match doc {doc_idx}, got doc {best_doc}. "
                f"Similarities: {doc_sims}"
            )


class TestMultiBatchAlignment:
    """Tests for alignment across multiple batches."""

    def test_alignment_with_small_batch_size(self, model):
        """
        Test alignment when documents are split across multiple batches.

        Using a small max_batch_tokens forces multiple batches.
        """
        docs = [
            "Cats are fluffy pets. They purr when happy.",
            "Dogs are loyal friends. They bark at strangers.",
            "Birds can fly high. They sing beautiful songs.",
            "Fish swim in water. They have colorful scales.",
        ]

        df, X = model.encode(
            docs,
            max_chunk_sents=[1, 2],
            chunk_overlap=0,
            max_batch_tokens=64,  # Force small batches
            show_progress=False,
        )

        # Verify alignment for each document
        queries = [
            (0, "cats fluffy pets purr"),
            (1, "dogs loyal bark"),
            (2, "birds fly sing"),
            (3, "fish swim water"),
        ]

        for doc_idx, query in queries:
            q = encode_text_as_query(model, query)

            # Get similarities for all segments
            sims_by_doc = {}
            for i in range(len(df)):
                d = df["document_idx"][i]
                if d not in sims_by_doc:
                    sims_by_doc[d] = []
                sims_by_doc[d].append(cosine_similarity(q, X[i]))

            # Target doc should have highest average similarity
            avg_sims = {d: np.mean(s) for d, s in sims_by_doc.items()}
            best_doc = max(avg_sims, key=avg_sims.get)

            assert best_doc == doc_idx, (
                f"Query '{query}' should match doc {doc_idx}, got doc {best_doc}. "
                f"Avg similarities: {avg_sims}"
            )

    def test_alignment_consistency_across_batch_sizes(self, model):
        """
        Verify that alignment is consistent regardless of batch size.

        The same documents should produce the same aligned results
        whether processed in 1 batch or multiple batches.
        """
        docs = [
            "Machine learning trains models on data.",
            "Web development creates interactive websites.",
        ]

        # Large batch (likely single batch)
        df_large, X_large = model.encode(
            docs,
            max_chunk_sents=[1],
            chunk_overlap=0,
            max_batch_tokens=4096,
            show_progress=False,
        )

        # Small batch (likely multiple batches)
        df_small, X_small = model.encode(
            docs,
            max_chunk_sents=[1],
            chunk_overlap=0,
            max_batch_tokens=64,
            show_progress=False,
        )

        # Same chunks in same order
        assert (
            df_large["chunk"].to_list() == df_small["chunk"].to_list()
        ), "Chunks should be identical regardless of batch size"

        # Same document assignments
        assert (
            df_large["document_idx"].to_list() == df_small["document_idx"].to_list()
        ), "Document assignments should be identical regardless of batch size"

        # Embeddings should be nearly identical (within numerical precision)
        np.testing.assert_array_almost_equal(
            X_large,
            X_small,
            decimal=4,
            err_msg="Embeddings should be identical regardless of batch size",
        )
