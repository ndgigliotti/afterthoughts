import numpy as np
import pandas as pd
import pyarrow as pa

from finephrase.utils import get_memory_report


class DocPhrases:
    """Conveniently store phrases and their embeddings for a set of documents.

    Parameters
    ----------
    doc_idx : np.ndarray
        An array of document indices corresponding to each phrase.
    phrases : pd.Series
        A pandas Series containing the phrases.
    embeds : np.ndarray
        An array of embeddings corresponding to each phrase.
    seq_idx : np.ndarray, optional
        An array of sequence indices corresponding to each phrase.
    batch_idx : np.ndarray, optional
        An array of batch indices corresponding to each phrase.

    Methods
    -------
    __getitem__(doc_idx)
        Returns the phrases for the given document index.
    _mask(doc_idx=None, *, seq_idx=None, batch_idx=None)
        Creates a mask for selecting phrases based on document, sequence, or batch indices.
    get_phrases(doc_idx=None, *, seq_idx=None, batch_idx=None)
        Returns the phrases for the given document, sequence, or batch indices.
    get_embeds(doc_idx=None, *, seq_idx=None, batch_idx=None)
        Returns the embeddings for the given document, sequence, or batch indices.
    __len__()
        Returns the number of unique documents.
    """

    def __init__(
        self,
        doc_idx,
        phrases,
        embeds,
        seq_idx=None,
        batch_idx=None,
        cluster_idx=None,
        phrases_format="dataframe",
    ):
        """
        Initialize the instance with document index, phrases, embeddings, sequence index, batch index, and cluster index.

        Parameters
        ----------
        doc_idx : int
            The index of the document.
        phrases : list
            A list of phrases associated with the document.
        embeds : list
            A list of embeddings corresponding to the phrases.
        seq_idx : int, optional
            The sequence index, by default None.
        batch_idx : int, optional
            The batch index, by default None.
        cluster_idx : int, optional
            The cluster index, by default None.
        phrases_format : str, optional
            The format in which to return the phrases, by default "list".
            Note: the phrases are stored as a PyArrow Array underneath.
            Options:
                - "list": A list of phrases.
                - "pyarrow": A PyArrow Array of phrases (fastest).
                - "numpy": A NumPy array of phrases (not recommended).
                - "series": A Pandas Series of phrases.
                - "dataframe": A DataFrame with a column for each index (default).
        """
        self.doc_idx = doc_idx
        self.phrases = phrases
        self.embeds = embeds
        self.seq_idx = seq_idx
        self.batch_idx = batch_idx
        self.cluster_idx = cluster_idx
        self.phrases_format = phrases_format
        self._validate()

    def _validate(self):
        """Validate the data to ensure that the lengths match and types are correct.

        Raises
        ------
        ValueError
            If the lengths of `doc_idx`, `phrases`, and `embeds` do not match.
            If the lengths of `doc_idx` and `seq_idx` do not match.
            If the lengths of `doc_idx` and `batch_idx` do not match.
            If the lengths of `doc_idx` and `cluster_idx` do not match.
            If `phrases` is not a PyArrow Array.
            If `phrases_format` is not one of 'list', 'pyarrow', 'numpy', 'series', or 'dataframe'.
        """
        if not isinstance(self.phrases, pa.Array):
            raise ValueError("Phrases must be a PyArrow Array.")
        if self.phrases_format not in [
            "list",
            "pyarrow",
            "numpy",
            "series",
            "dataframe",
        ]:
            raise ValueError(
                "Invalid `phrases_format`. Must be one of 'list', 'pyarrow', 'numpy', 'series', or 'dataframe'."
            )
        if len(self.doc_idx) != len(self.phrases):
            raise ValueError("Length of `doc_idx` and `phrases` must match.")
        if len(self.doc_idx) != len(self.embeds):
            raise ValueError("Length of `doc_idx` and `embeds` must match.")
        if self.seq_idx is not None and len(self.doc_idx) != len(self.seq_idx):
            raise ValueError("Length of `doc_idx` and `seq_idx` must match.")
        if self.batch_idx is not None and len(self.doc_idx) != len(self.batch_idx):
            raise ValueError("Length of `doc_idx` and `batch_idx` must match.")
        if self.cluster_idx is not None and len(self.doc_idx) != len(self.cluster_idx):
            raise ValueError("Length of `doc_idx` and `cluster_idx` must match.")

    def __getitem__(self, doc_idx):
        """Retrieve phrases for a given document index.

        Parameters
        ----------
        doc_idx : int, slice, or np.ndarray
            The index of the document to retrieve phrases for.

        Returns
        -------
        list
            A list of phrases associated with the specified document index.
        """
        return self.get_phrases(doc_idx)

    def set_cluster_idx(self, cluster_idx):
        self.cluster_idx = cluster_idx
        self._validate()

    @property
    def data(self):
        """Return the data in the instance."""
        data = {"doc_idx": self.doc_idx, "phrases": self.phrases, "embeds": self.embeds}
        if self.seq_idx is not None:
            data["seq_idx"] = self.seq_idx
        if self.batch_idx is not None:
            data["batch_idx"] = self.batch_idx
        return data

    @property
    def num_docs(self):
        """Return the number of unique documents."""
        return len(self)

    @property
    def num_phrases(self):
        """Return the number of phrases in the dataset."""
        return len(self.phrases)

    @property
    def phrase_idx(self):
        """Return the index of the phrases."""
        return np.arange(len(self.phrases))

    def _get_repr_df(self, num=12):
        """Return a DataFrame representation of the instance."""
        if num % 2 != 0:
            raise ValueError("Number of phrases must be even.")
        half = num // 2
        phrase_idx_head = np.arange(min(half, len(self.phrases)))
        phrase_idx_tail = (-phrase_idx_head[::-1]) + len(self.phrases) - 1
        phrase_idx = np.unique(np.hstack([phrase_idx_head, phrase_idx_tail]))
        df = {
            "doc_idx": self.doc_idx[phrase_idx],
            "phrase": self.phrases.take(phrase_idx),
        }
        return pd.DataFrame(df, index=phrase_idx)

    def __repr__(self):
        """Return a string representation of the instance."""
        df = self._get_repr_df(num=12)
        title = f"DocPhrases(num_docs={self.num_docs}, num_phrases={self.num_phrases}, embeds={self.embeds.shape})"
        string = f"{df.to_string(max_rows=10)}\n\n{title}"
        return string

    def __str__(self):
        """Return a string representation of the instance."""
        return repr(self)

    def _mask(self, doc_idx=None, *, seq_idx=None, batch_idx=None, cluster_idx=None):
        """Generate a mask based on specified index criteria.

        Parameters
        ----------
        doc_idx : int, slice, array-like, or None, optional
            Index or indices to select based on document indices. Default is None.
        seq_idx : int, slice, array-like, or None, optional
            Index or indices to select based on sequence indices. Default is None.
        batch_idx : int, slice, array-like, or None, optional
            Index or indices to select based on batch indices. Default is None.
        cluster_idx : int, slice, array-like, or None, optional
            Index or indices to select based on cluster indices. Default is None.

        Returns
        -------
        mask : np.ndarray
            Boolean array where selected indices are marked as True.

        Raises
        ------
        ValueError
            If more than one of `doc_idx`, `seq_idx`, `batch_idx`, or `cluster_idx` is specified.
            If sequence indices are unavailable when `seq_idx` is specified.
            If batch indices are unavailable when `batch_idx` is specified.
            If cluster indices are unavailable when `cluster_idx` is specified.
            If a boolean mask does not have the same length as the reference index.
        """
        # Check that at most one of doc_idx, seq_idx, batch_idx, or cluster_idx is not None
        if sum(x is not None for x in (doc_idx, seq_idx, batch_idx, cluster_idx)) > 1:
            raise ValueError(
                "Can only specify one of `doc_idx`, `seq_idx`, `batch_idx`, or `cluster_idx`."
            )
        # Determine which index to use as reference
        if doc_idx is not None:
            ref_idx = self.doc_idx
            sel_idx = doc_idx
        elif seq_idx is not None:
            if self.seq_idx is None:
                raise ValueError("Sequence indices are unavailable.")
            ref_idx = self.seq_idx
            sel_idx = seq_idx
        elif batch_idx is not None:
            if self.batch_idx is None:
                raise ValueError("Batch indices are unavailable.")
            ref_idx = self.batch_idx
            sel_idx = batch_idx
        elif cluster_idx is not None:
            if self.cluster_idx is None:
                raise ValueError("Cluster indices are unavailable.")
            ref_idx = self.cluster_idx
            sel_idx = cluster_idx
        else:
            sel_idx = None
        if isinstance(sel_idx, slice):
            ref_len = np.unique(ref_idx).shape[0]
            sel_idx = np.arange(*sel_idx.indices(ref_len))
        if sel_idx is None:
            # Select all
            mask = np.ones_like(self.doc_idx, dtype=bool)
        elif isinstance(sel_idx, np.ndarray) and sel_idx.dtype == bool:
            if sel_idx.shape[0] != ref_idx.shape[0]:
                raise ValueError(
                    "Boolean mask must have the same length as the reference index."
                )
            mask = sel_idx
        elif isinstance(sel_idx, int):
            mask = ref_idx == sel_idx
        else:
            mask = np.isin(ref_idx, sel_idx)
        return mask

    def get_phrases(
        self, doc_idx=None, *, seq_idx=None, batch_idx=None, cluster_idx=None
    ):
        """
        Retrieve phrases from the document based on the specified indices.

        Parameters
        ----------
        doc_idx : int, optional
            The index of the document to retrieve phrases from. If None, all documents are considered.
        seq_idx : int, optional
            The index of the sequence within the document to retrieve phrases from. If None, all sequences are considered.
        batch_idx : int, optional
            The index of the batch within the document to retrieve phrases from. If None, all batches are considered.
        cluster_idx : int, optional
            The index of the cluster within the document to retrieve phrases from. If None, all clusters are considered.

        Returns
        -------
        list
            A list of phrases that match the specified indices.
        """
        mask = self._mask(
            doc_idx, seq_idx=seq_idx, batch_idx=batch_idx, cluster_idx=cluster_idx
        )
        results = self.phrases.filter(mask)
        # Convert results from PyArrow to the specified format
        if self.phrases_format == "list":
            results = results.to_pylist()
        elif self.phrases_format == "numpy":
            results = results.to_numpy()
        elif self.phrases_format == "series":
            results = results.to_pandas()
            results.name = "phrase"
        elif self.phrases_format == "dataframe":
            df = {"phrase_idx": np.nonzero(mask)[0], "doc_idx": self.doc_idx[mask]}
            if self.seq_idx is not None:
                df["seq_idx"] = self.seq_idx[mask]
            if self.batch_idx is not None:
                df["batch_idx"] = self.batch_idx[mask]
            if self.cluster_idx is not None:
                df["cluster_idx"] = self.cluster_idx[mask]
            df["phrase"] = results
            results = pd.DataFrame(df)
        return results

    def get_embeds(
        self, doc_idx=None, *, seq_idx=None, batch_idx=None, cluster_idx=None
    ):
        """
        Retrieve embeddings based on specified indices.

        Parameters
        ----------
        doc_idx : int, optional
            The document index to filter embeddings. If None, no filtering is applied based on document index.
        seq_idx : int, optional
            The sequence index to filter embeddings. If None, no filtering is applied based on sequence index.
        batch_idx : int, optional
            The batch index to filter embeddings. If None, no filtering is applied based on batch index.
        cluster_idx : int, optional
            The cluster index to filter embeddings. If None, no filtering is applied based on cluster index.

        Returns
        -------
        numpy.ndarray
            The filtered embeddings based on the provided indices.
        """
        mask = self._mask(
            doc_idx, seq_idx=seq_idx, batch_idx=batch_idx, cluster_idx=cluster_idx
        )
        return self.embeds[mask]

    def __len__(self):
        """
        Returns the number of unique documents.

        Returns
        -------
        int
            The number of unique documents in the dataset.
        """
        return np.unique(self.doc_idx).shape[0]

    def memory_usage(self, readable=True):
        """Return the memory usage of the instance."""
        return get_memory_report(self.data, readable=readable)
