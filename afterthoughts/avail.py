# Copyright 2024-2026 Nicholas Gigliotti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy imports and validation for optional dependencies.

This module provides functions for importing optional dependencies with
helpful error messages when they are not installed. It enables graceful
degradation when optional features (pandas support, alternative sentence
tokenizers) are unavailable.

Functions
---------
get_pandas : Attempt to import pandas, return None if unavailable
require_pandas : Import pandas or raise informative ImportError
require_nltk : Import NLTK and ensure punkt data is downloaded
require_pysbd : Import pysbd.Segmenter or raise informative ImportError
require_syntok : Import syntok.segmenter.analyze or raise informative ImportError

Notes
-----
All `require_*` functions raise ImportError with installation instructions
if the corresponding package is not available. The `get_*` functions return
None instead of raising errors.
"""

from typing import Any


def get_pandas() -> Any | None:
    """Attempt to import pandas module.

    Returns
    -------
    module or None
        The pandas module if available, None otherwise.

    Examples
    --------
    >>> pd = get_pandas()
    >>> if pd is not None:
    ...     df = pd.DataFrame({"a": [1, 2, 3]})
    """
    try:
        import pandas

        return pandas
    except ImportError:
        return None


def require_pandas() -> Any:
    """Import pandas module or raise informative error.

    Returns
    -------
    module
        The pandas module.

    Raises
    ------
    ImportError
        If pandas is not installed, with installation instructions.

    Examples
    --------
    >>> pd = require_pandas()  # doctest: +SKIP
    >>> df = pd.DataFrame({"a": [1, 2, 3]})  # doctest: +SKIP
    """
    pd = get_pandas()
    if pd is None:
        raise ImportError(
            "pandas is required for return_frame='pandas'. Install it with: pip install pandas"
        )
    return pd


def require_syntok() -> Any:
    """Import syntok.segmenter.analyze function or raise informative error.

    Returns
    -------
    function
        The syntok.segmenter.analyze function for sentence segmentation.

    Raises
    ------
    ImportError
        If syntok is not installed, with installation instructions.

    Notes
    -----
    syntok is a sentence and token segmentation library that provides
    rule-based segmentation with support for abbreviations and edge cases.
    """
    try:
        from syntok.segmenter import analyze

        return analyze
    except ImportError:
        raise ImportError(
            "syntok is required for method='syntok'. Install it with: pip install syntok"
        ) from None


def require_nltk() -> Any:
    """Import NLTK module and ensure punkt data is available.

    This function imports the NLTK library and automatically downloads
    the punkt sentence tokenizer data if it is not already present.

    Returns
    -------
    module
        The nltk module.

    Raises
    ------
    ImportError
        If NLTK is not installed, with installation instructions.

    Notes
    -----
    The punkt tokenizer is a pre-trained model for sentence tokenization
    that uses an unsupervised algorithm to learn abbreviation detection
    and sentence boundary detection rules. If punkt data is not found,
    it will be downloaded automatically on first use.
    """
    try:
        import nltk
    except ImportError:
        raise ImportError(
            "nltk is required for method='nltk'. Install it with: pip install nltk"
        ) from None

    # Check if punkt is installed, download if needed
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    return nltk


def require_pysbd() -> Any:
    """Import pysbd.Segmenter class or raise informative error.

    Returns
    -------
    class
        The pysbd.Segmenter class for sentence segmentation.

    Raises
    ------
    ImportError
        If pysbd is not installed, with installation instructions.

    Notes
    -----
    pysbd (Python Sentence Boundary Disambiguation) is designed to handle
    edge cases like abbreviations (Dr., U.S., etc.), numbers, and other
    punctuation more accurately than simpler rule-based tokenizers. It is
    based on the Pragmatic Segmenter Ruby library.
    """
    try:
        from pysbd import Segmenter

        return Segmenter
    except ImportError:
        raise ImportError(
            "pysbd is required for method='pysbd'. Install it with: pip install pysbd"
        ) from None
