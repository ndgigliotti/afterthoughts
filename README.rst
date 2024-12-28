==========
Grammatron
==========


.. .. image:: https://img.shields.io/pypi/v/grammatron.svg
..         :target: https://pypi.python.org/pypi/grammatron

.. .. image:: https://img.shields.io/travis/ndgigliotti/grammatron.svg
..         :target: https://travis-ci.com/ndgigliotti/grammatron

.. .. image:: https://readthedocs.org/projects/grammatron/badge/?version=latest
..         :target: https://embed-gram.readthedocs.io/en/latest/?version=latest
..         :alt: Documentation Status




Fast n-gram embeddings using transformers.


* Free software: Apache Software License 2.0

This is a new project that is heavily under development. Please check back soon for updates.

Concept
-------

The idea behind this project is to provide a fast and context-aware method of embedding n-grams using transformers.
Sometimes to obtain n-gram embeddings, data scientists extract n-grams upstream of the model and run each n-gram through the model as a separate sequence.
This is slow and fails to incorporate the context surrounding each n-gram. A better approach is to run the entire document 
through the model and derive the n-gram embeddings from the token embeddings in the final hidden state. Running the entire document through the 
model not only takes advantage of the transformer's ability to process each token in parallel, but also incorporates context into the n-gram representations.
The final representations are derived by finding every possible token n-gram and, for each, averaging the corresponding token embeddings.


To-Do
--------

* Support input chunking for long sequences
* Add features for handling large datasets

License
-------

This project is licensed under the Apache License 2.0. 

Copyright 2024 Nicholas Gigliotti.

You may use, distribute, and modify this project under the terms of the Apache License 2.0. For detailed information, see the `LICENSE <LICENSE>`_ file included in this repository or visit the official `Apache License website <http://www.apache.org/licenses/LICENSE-2.0>`_.
