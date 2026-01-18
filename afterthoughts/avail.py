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

"""Lazy imports for optional dependencies."""

from typing import Any


def get_pandas() -> Any | None:
    """Return pandas module or None if not available."""
    try:
        import pandas

        return pandas
    except ImportError:
        return None


def require_pandas() -> Any:
    """Return pandas module or raise ImportError if not available."""
    pd = get_pandas()
    if pd is None:
        raise ImportError(
            "pandas is required for return_frame='pandas'. Install it with: pip install pandas"
        )
    return pd


def require_syntok() -> Any:
    """Return syntok.segmenter.analyze or raise ImportError if not available."""
    try:
        from syntok.segmenter import analyze

        return analyze
    except ImportError:
        raise ImportError(
            "syntok is required for method='syntok'. Install it with: pip install syntok"
        ) from None


def require_nltk() -> Any:
    """Return nltk module or raise ImportError if not available.

    Also ensures punkt tokenizer data is downloaded.
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
    """Return pysbd.Segmenter class or raise ImportError if not available."""
    try:
        from pysbd import Segmenter

        return Segmenter
    except ImportError:
        raise ImportError(
            "pysbd is required for method='pysbd'. Install it with: pip install pysbd"
        ) from None
