try:
    import pandas as pd

    _HAS_PANDAS = True
    del pd
except ImportError:
    _HAS_PANDAS = False

try:
    import sklearn

    _HAS_SKLEARN = True
    del sklearn
except ImportError:
    _HAS_SKLEARN = False

try:
    import faiss

    _HAS_FAISS = True
    del faiss
except ImportError:
    _HAS_FAISS = False
