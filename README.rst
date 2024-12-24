==========
Grammatron
==========


.. image:: https://img.shields.io/pypi/v/grammatron.svg
        :target: https://pypi.python.org/pypi/grammatron

.. image:: https://img.shields.io/travis/ndgigliotti/grammatron.svg
        :target: https://travis-ci.com/ndgigliotti/grammatron

.. image:: https://readthedocs.org/projects/grammatron/badge/?version=latest
        :target: https://embed-gram.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Fast n-gram embeddings using transformers.


* Free software: MIT license
.. * Documentation: https://grammatron.readthedocs.io.

This is a new project that is heavily under development. Please check back soon for updates.

Concept
-------

The idea behind this project is to provide a fast and context-aware method of embedding n-grams using transformers.
Sometimes to obtain n-gram embeddings, data scientists extract n-grams upstream of the model and run each n-gram through the model as a separate sequence.
This is slow and fails to incorporate the context surrounding each n-gram. A better approach is to run the entire document 
through the model and derive the n-gram embeddings from the token embeddings in the final hidden state. Running the entire document through the 
model not only takes advantage of the transformer's ability to process each token in parallel, but also incorporates context into the n-gram representations.
The final representations are derived by finding every possible token n-gram and, for each, averaging the corresponding token embeddings.


Features
--------

* TODO