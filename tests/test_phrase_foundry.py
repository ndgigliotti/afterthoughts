# Copyright 2024 Nicholas Gigliotti
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

"""Tests for `Phrase Foundry` package."""

import re
import numpy as np
import torch
import pytest
from phrase_foundry.phrase_foundry import TokenizedDataset, PhraseFoundry, get_ngram_idx


def test_tokenized_dataset_init_valid_inputs():
    """
    Test the initialization of TokenizedDataset with valid inputs.

    This test verifies that the TokenizedDataset class correctly initializes
    when provided with valid input data. It checks that the 'inputs' attribute
    of the dataset instance is set to the provided input data.

    Inputs:
    - A dictionary containing a single key 'input_ids' with a tensor of shape (2, 3).

    Assertions:
    - The 'inputs' attribute of the TokenizedDataset instance should be the same
        as the provided input data.
    """
    inputs = {"input_ids": torch.Tensor([[1, 2, 3], [4, 5, 6]])}
    dataset = TokenizedDataset(inputs)
    assert dataset.inputs is inputs


def test_tokenized_dataset_init_invalid_input_ids_type():
    """
    Test the initialization of TokenizedDataset with invalid input_ids type.

    This test verifies that a TypeError is raised when the `input_ids` provided
    to the TokenizedDataset constructor is not a torch.Tensor. The expected
    error message is "`input_ids` must be a torch.Tensor."

    Raises:
        TypeError: If `input_ids` is not a torch.Tensor.
    """
    inputs = {"input_ids": [[1, 2, 3], [4, 5, 6]]}
    with pytest.raises(TypeError, match="`input_ids` must be a torch.Tensor."):
        TokenizedDataset(inputs)


def test_tokenized_dataset_init_missing_input_ids():
    """
    Test the initialization of the TokenizedDataset class when the 'input_ids' key is missing from the input dictionary.

    This test ensures that a KeyError is raised when the 'input_ids' key is not present in the input dictionary.

    Raises:
        KeyError: If the 'input_ids' key is missing from the input dictionary.
    """
    inputs = {"attention_mask": torch.Tensor([[1, 1, 1], [1, 1, 1]])}
    with pytest.raises(KeyError):
        TokenizedDataset(inputs)


def test_tokenized_dataset_len():
    """
    Test the length of the TokenizedDataset.

    This test verifies that the length of the TokenizedDataset is correctly
    calculated based on the input tensor. It creates a TokenizedDataset
    instance with a given input tensor and asserts that the length of the
    dataset matches the expected value.

    Assertions:
        - The length of the dataset should be 2.
    """
    inputs = {"input_ids": torch.Tensor([[1, 2, 3], [4, 5, 6]])}
    dataset = TokenizedDataset(inputs)
    assert len(dataset) == 2


def test_tokenized_dataset_getitem():
    """
    Test the __getitem__ method of the TokenizedDataset class.

    This test verifies that the TokenizedDataset class correctly returns the
    expected input_ids and attention_mask for a given index.

    The test creates a TokenizedDataset instance with predefined input_ids and
    attention_mask tensors, retrieves the first item, and asserts that the
    returned input_ids and attention_mask match the expected values.

    Assertions:
        - The input_ids of the first item should be [1, 2, 3].
        - The attention_mask of the first item should be [1, 1, 1].
    """
    inputs = {
        "input_ids": torch.Tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.Tensor([[1, 1, 1], [1, 1, 1]]),
    }
    dataset = TokenizedDataset(inputs)
    item = dataset[0]
    assert item["input_ids"].tolist() == [1, 2, 3]
    assert item["attention_mask"].tolist() == [1, 1, 1]


def test_tokenized_dataset_getitem_out_of_bounds():
    """
    Test that accessing an out-of-bounds index in the TokenizedDataset raises an IndexError.

    This test creates a TokenizedDataset with a sample input tensor and verifies that
    attempting to access an index that is out of the dataset's bounds raises an IndexError.

    Raises:
        IndexError: If an out-of-bounds index is accessed.
    """
    inputs = {"input_ids": torch.Tensor([[1, 2, 3], [4, 5, 6]])}
    dataset = TokenizedDataset(inputs)
    with pytest.raises(IndexError):
        dataset[2]


def test_tokenized_dataset_exclude():
    """
    Test the exclusion of 'overflow_to_sample_mapping' from the tokenized dataset.

    This test verifies that the 'overflow_to_sample_mapping' key is not present in the
    items of the TokenizedDataset. It creates a sample input dictionary with 'input_ids'
    and 'overflow_to_sample_mapping', initializes the TokenizedDataset with this input,
    and checks that the 'overflow_to_sample_mapping' key is excluded from the dataset items.

    Assertions:
        - The 'overflow_to_sample_mapping' key should not be present in the dataset item.
    """
    inputs = {
        "input_ids": torch.Tensor([[1, 2, 3], [4, 5, 6]]),
        "overflow_to_sample_mapping": torch.Tensor([0, 1]),
    }
    dataset = TokenizedDataset(inputs)
    item = dataset[0]
    assert "overflow_to_sample_mapping" not in item


def test_get_ngram_idx_valid_input_tensor():
    """
    Test the `get_ngram_idx` function with valid input tensor.

    This test checks if the `get_ngram_idx` function correctly generates n-gram indices
    for a given input tensor and n-gram range. The input tensor is a 2D tensor with a
    single row of integers. The expected output is a list of numpy arrays representing
    the indices of the n-grams for the specified n-gram range.

    Tested function:
    - `get_ngram_idx(input_ids, ngram_range)`

    Expected output:
    - A list of numpy arrays where each array contains the indices of the n-grams
        for the specified n-gram range.

    Assertions:
    - The length of the output list should match the length of the expected output list.
    - Each element in the output list should be equal to the corresponding element
        in the expected output list.
    """
    input_ids = torch.Tensor([[1, 2, 3, 4, 5, 6]])
    ngram_range = (2, 3)
    expected_output = [
        np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]),
        np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]),
    ]
    output = get_ngram_idx(input_ids, ngram_range)
    assert len(output) == len(expected_output)
    for out, exp in zip(output, expected_output):
        np.testing.assert_array_equal(out, exp)


def test_get_ngram_idx_valid_input_ndarray():
    """
    Test the `get_ngram_idx` function with valid input ndarray.

    This test checks if the `get_ngram_idx` function correctly generates n-gram indices
    for a given input array and n-gram range. The input array is a 2D numpy array, and
    the n-gram range specifies the range of n-grams to generate.

    Test case:
    - input_ids: A 2D numpy array with shape (1, 6) containing the sequence [1, 2, 3, 4, 5, 6].
    - ngram_range: A tuple (2, 3) specifying the range of n-grams to generate.

    Expected output:
    - A list of numpy arrays containing the n-gram indices for the specified range:
        [
                np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])

    The test asserts that the length of the output matches the expected output and
    that each element in the output matches the corresponding element in the expected output.
    """
    input_ids = np.array([[1, 2, 3, 4, 5, 6]])
    ngram_range = (2, 3)
    expected_output = [
        np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]),
        np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]),
    ]
    output = get_ngram_idx(input_ids, ngram_range)
    assert len(output) == len(expected_output)
    for out, exp in zip(output, expected_output):
        np.testing.assert_array_equal(out, exp)


def test_get_ngram_idx_empty_input():
    """
    Test the `get_ngram_idx` function with an empty input.

    This test case checks the behavior of the `get_ngram_idx` function when provided
    with an empty input array. It ensures that the function returns an empty list
    as expected when there are no elements to form n-grams.

    Tested function:
    - get_ngram_idx(input_ids, ngram_range)

    Test case:
    - input_ids: An empty 2D numpy array.
    - ngram_range: A tuple specifying the range of n-grams to generate (2, 3).
    - expected_output: An empty list, as no n-grams can be formed from an empty input.

    Assertions:
    - The output of the `get_ngram_idx` function should match the expected output.
    """
    input_ids = np.array([[]])
    ngram_range = (2, 3)
    expected_output = []
    output = get_ngram_idx(input_ids, ngram_range)
    assert output == expected_output


def test_get_ngram_idx_single_element():
    """
    Test the `get_ngram_idx` function with a single element input.

    This test checks the behavior of the `get_ngram_idx` function when provided with
    an input array containing a single element and an n-gram range of (1, 1).

    Test case:
    - input_ids: A numpy array with a single element [[1]]
    - ngram_range: A tuple representing the range of n-grams to generate (1, 1)
    - expected_output: A list containing a numpy array with a single element [[0]]

    The test verifies that the output of the `get_ngram_idx` function matches the
    expected output by comparing the length of the output list and the contents of
    each array within the list using numpy's `assert_array_equal` method.
    """
    input_ids = np.array([[1]])
    ngram_range = (1, 1)
    expected_output = [np.array([[0]])]
    output = get_ngram_idx(input_ids, ngram_range)
    assert len(output) == len(expected_output)
    for out, exp in zip(output, expected_output):
        np.testing.assert_array_equal(out, exp)


def test_get_ngram_idx_ngram_range_out_of_bounds():
    """
    Test the `get_ngram_idx` function with an n-gram range that is out of bounds.

    This test checks the behavior of the `get_ngram_idx` function when the specified
    n-gram range is larger than the length of the input sequence. The function is
    expected to return an empty list in this case.

    Test case:
    - input_ids: A numpy array with a single sequence [1, 2, 3].
    - ngram_range: A tuple (4, 5) specifying the range of n-grams to extract.
    - expected_output: An empty list, as no n-grams of length 4 or 5 can be formed
        from the input sequence.

    Asserts:
    - The output of the `get_ngram_idx` function matches the expected output.
    """
    input_ids = np.array([[1, 2, 3]])
    ngram_range = (4, 5)
    expected_output = []
    output = get_ngram_idx(input_ids, ngram_range)
    assert output == expected_output


def test_phrase_foundry_init_valid_model_name():
    """
    Test the initialization of the PhraseFoundry class with a valid model name.

    This test verifies that the PhraseFoundry class is correctly initialized with
    the provided model name. It checks that the tokenizer and model within the
    PhraseFoundry instance are set to the correct paths, and that the default
    values for `invalid_start_token_pattern` and `exclude_tokens` are set as
    expected.

    Assertions:
        - The tokenizer's `name_or_path` attribute matches the provided model name.
        - The model's `name_or_path` attribute matches the provided model name.
        - The `invalid_start_token_pattern` attribute is set to the default value "^##".
        - The `exclude_tokens` attribute is set to None.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name)
    assert phrase_foundry.tokenizer.name_or_path == model_name
    assert phrase_foundry.model.name_or_path == model_name
    assert phrase_foundry.invalid_start_token_pattern == r"^##"
    assert phrase_foundry.exclude_tokens is None


def test_phrase_foundry_init_invalid_model_name():
    """
    Test the initialization of the PhraseFoundry class with an invalid model name.

    This test verifies that an OSError is raised when attempting to initialize
    the PhraseFoundry class with a model name that does not exist or is invalid.

    Raises:
        OSError: If the model name provided is invalid.
    """
    model_name = "invalid-model-name"
    with pytest.raises(OSError):
        PhraseFoundry(model_name)


def test_phrase_foundry_init_custom_device():
    """
    Test the initialization of the PhraseFoundry class with a custom device.

    This test verifies that the PhraseFoundry class can be initialized with a
    specified model name and device, and checks that the device attribute of
    the PhraseFoundry instance is correctly set to the specified device.

    Assertions:
        - The device type of the PhraseFoundry instance should be "cpu".
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    device = "cpu"
    phrase_foundry = PhraseFoundry(model_name, device=device)
    assert phrase_foundry.device.type == "cpu"


def test_phrase_foundry_init_custom_invalid_start_token_pattern():
    """
    Test the initialization of the PhraseFoundry class with a custom invalid start token pattern.

    This test verifies that the PhraseFoundry instance correctly initializes with a given
    invalid start token pattern and that the corresponding invalid start token IDs are set
    as expected.

    Steps:
    1. Define the model name and an invalid start token pattern.
    2. Initialize the PhraseFoundry instance with the provided model name and invalid start token pattern.
    3. Verify that the invalid start token pattern is correctly set in the PhraseFoundry instance.
    4. Verify that the invalid start token IDs match the expected values.

    Assertions:
    - The invalid start token pattern in the PhraseFoundry instance should match the provided pattern.
    - The invalid start token IDs in the PhraseFoundry instance should match the expected IDs.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    invalid_start_token_pattern = r"^@"
    phrase_foundry = PhraseFoundry(
        model_name, invalid_start_token_pattern=invalid_start_token_pattern
    )
    expected_ids = [1030]
    assert phrase_foundry.invalid_start_token_pattern == invalid_start_token_pattern
    np.testing.assert_array_equal(phrase_foundry.invalid_start_token_ids, expected_ids)


def test_phrase_foundry_init_custom_exclude_tokens():
    """
    Test the initialization of the PhraseFoundry class with custom exclude tokens.

    This test verifies that the PhraseFoundry class is correctly initialized with
    a specified model name and a list of tokens to exclude. It checks that the
    exclude_tokens attribute of the PhraseFoundry instance matches the provided
    exclude_tokens list.

    Assertions:
        - The exclude_tokens attribute of the PhraseFoundry instance should be equal
          to the provided exclude_tokens list.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    exclude_tokens = ["[CLS]", "[SEP]"]
    phrase_foundry = PhraseFoundry(model_name, exclude_tokens=exclude_tokens)
    assert phrase_foundry.exclude_tokens == exclude_tokens


def test_phrase_foundry_to_cpu():
    """
    Test the `to` method of the `PhraseFoundry` class to ensure that the model is moved to the CPU.

    This test initializes a `PhraseFoundry` instance with a specified model name and sets the device to "cuda".
    It then calls the `to` method to move the model to the CPU and asserts that the device type is "cpu".

    Raises:
        AssertionError: If the device type is not "cpu" after calling the `to` method.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name, device="cuda")
    phrase_foundry.to("cpu")
    assert phrase_foundry.device.type == "cpu"


def test_phrase_foundry_to_cuda():
    """
    Test the `to` method of the `PhraseFoundry` class to ensure that the model is correctly moved to the CUDA device.

    This test initializes a `PhraseFoundry` instance with a specified model name and sets the device to "cpu".
    It then moves the model to the "cuda" device and asserts that the device type of the `PhraseFoundry` instance is "cuda".

    Raises:
        AssertionError: If the device type of the `PhraseFoundry` instance is not "cuda".
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name, device="cpu")
    phrase_foundry.to("cuda")
    assert phrase_foundry.device.type == "cuda"


def test_phrase_foundry_to_invalid_device():
    """
    Test the PhraseFoundry model's behavior when attempting to move it to an invalid device.

    This test initializes a PhraseFoundry instance with a specified model name and sets the device to "cpu".
    It then attempts to move the model to an invalid device and expects a RuntimeError to be raised.

    Raises:
        RuntimeError: If the model is moved to an invalid device.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name, device="cpu")
    with pytest.raises(RuntimeError):
        phrase_foundry.to("invalid_device")


def test_phrase_foundry_exclude_token_ids_none():
    """
    Test the `exclude_token_ids` attribute of the `PhraseFoundry` class when no token IDs are excluded.

    This test initializes a `PhraseFoundry` instance with a specified model name and checks that the
    `exclude_token_ids` attribute matches the `all_special_ids` attribute of the tokenizer.

    Assertions:
        - The `exclude_token_ids` attribute should be equal to the `all_special_ids` attribute of the tokenizer.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name)
    expected_ids = phrase_foundry.tokenizer.all_special_ids
    np.testing.assert_array_equal(phrase_foundry.exclude_token_ids, expected_ids)


def test_phrase_foundry_exclude_token_ids_str_list():
    """
    Test the PhraseFoundry class to ensure that the exclude_token_ids attribute is correctly set
    when a list of token strings is provided.

    This test initializes a PhraseFoundry instance with a specified model and a list of tokens
    to exclude. It then verifies that the exclude_token_ids attribute of the PhraseFoundry instance
    matches the expected token IDs converted from the provided token strings.

    Assertions:
        - The exclude_token_ids attribute of the PhraseFoundry instance should match the expected
          token IDs converted from the provided token strings.

    Raises:
        AssertionError: If the exclude_token_ids attribute does not match the expected token IDs.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    exclude_tokens = ["[CLS]", "[SEP]"]
    phrase_foundry = PhraseFoundry(model_name, exclude_tokens=exclude_tokens)
    expected_ids = phrase_foundry.tokenizer.convert_tokens_to_ids(exclude_tokens)
    np.testing.assert_array_equal(phrase_foundry.exclude_token_ids, expected_ids)


def test_phrase_foundry_exclude_token_ids_int_list():
    """
    Test the PhraseFoundry class to ensure that the exclude_token_ids attribute is correctly set
    when provided with a list of integer token IDs.

    This test initializes a PhraseFoundry instance with a specified model name and a list of
    token IDs to exclude. It then asserts that the exclude_token_ids attribute of the instance
    matches the provided list of token IDs.

    Tested attributes:
    - model_name: The name of the model to be used by the PhraseFoundry instance.
    - exclude_tokens: A list of integer token IDs to be excluded.
    - phrase_foundry.exclude_token_ids: The attribute that should match the exclude_tokens list.

    Assertions:
    - The exclude_token_ids attribute of the PhraseFoundry instance should be equal to the
        exclude_tokens list.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    exclude_tokens = [101, 102]
    phrase_foundry = PhraseFoundry(model_name, exclude_tokens=exclude_tokens)
    np.testing.assert_array_equal(phrase_foundry.exclude_token_ids, exclude_tokens)


def test_phrase_foundry_exclude_token_ids_invalid_type():
    """
    Test that PhraseFoundry raises a TypeError when `exclude_tokens` is not a list.

    This test verifies that the PhraseFoundry class correctly raises a TypeError
    when the `exclude_tokens` parameter is provided as a string instead of a list.
    The expected error message should indicate that `exclude_tokens` must be a
    list of token IDs or tokens.

    Raises:
        TypeError: If `exclude_tokens` is not a list.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    exclude_tokens = "[CLS]"
    with pytest.raises(
        TypeError, match="`exclude_tokens` must be a list of token IDs or tokens."
    ):
        PhraseFoundry(model_name, exclude_tokens=exclude_tokens)


def test_phrase_foundry_invalid_start_token_ids():
    """
    Test the PhraseFoundry class for handling invalid start token IDs.

    This test initializes a PhraseFoundry instance with a specific model and
    invalid start token pattern. It then verifies that the invalid start token
    IDs in the PhraseFoundry instance match the expected token IDs derived from
    the tokenizer's vocabulary.

    The test checks:
    - The model name used for the PhraseFoundry instance.
    - The pattern used to identify invalid start tokens.
    - The expected invalid start token IDs based on the pattern.
    - The actual invalid start token IDs in the PhraseFoundry instance.

    Assertions:
    - The invalid start token IDs in the PhraseFoundry instance should match
      the expected token IDs.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    invalid_start_token_pattern = r"^##"
    phrase_foundry = PhraseFoundry(
        model_name, invalid_start_token_pattern=invalid_start_token_pattern
    )
    expected_ids = [
        v
        for k, v in phrase_foundry.tokenizer.vocab.items()
        if re.search(invalid_start_token_pattern, k)
    ]
    np.testing.assert_array_equal(phrase_foundry.invalid_start_token_ids, expected_ids)


def test_phrase_foundry_invalid_start_token_ids_none():
    """
    Test the PhraseFoundry class with an invalid start token pattern set to None.

    This test verifies that when the invalid_start_token_pattern is set to None,
    the PhraseFoundry instance initializes with an empty list for invalid_start_token_ids.

    Steps:
    1. Initialize a PhraseFoundry instance with a specified model name and invalid_start_token_pattern set to None.
    2. Define the expected invalid_start_token_ids as an empty list.
    3. Assert that the invalid_start_token_ids attribute of the PhraseFoundry instance matches the expected empty list.

    Expected Result:
    The invalid_start_token_ids attribute should be an empty list.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name, invalid_start_token_pattern=None)
    expected_ids = []
    np.testing.assert_array_equal(phrase_foundry.invalid_start_token_ids, expected_ids)


def test_phrase_foundry_extract_ngrams():
    """
    Test the `extract_ngrams` method of the `PhraseFoundry` class.

    This test verifies that the `extract_ngrams` method correctly extracts n-grams
    from the input token IDs and token embeddings, and that the resulting n-grams,
    sequence indices, and n-gram vectors have the expected properties.

    The test uses the following parameters:
    - `model_name`: The name of the model to be used by `PhraseFoundry`.
    - `input_ids`: A numpy array representing the token IDs of the input sequence.
    - `token_embeds`: A numpy array representing the token embeddings of the input sequence.
    - `ngram_range`: A tuple specifying the range of n-grams to be extracted.

    Assertions:
    - The lengths of `seq_idx`, `ngrams`, and `ngram_vecs` should be equal.
    - The second dimension of `ngram_vecs` should be 384, which corresponds to the embedding size.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name)
    input_ids = np.array([[101, 2003, 1037, 2742, 102]])
    token_embeds = np.random.rand(1, 5, 384)
    ngram_range = (2, 3)
    seq_idx, ngrams, ngram_vecs = phrase_foundry.extract_ngrams(
        input_ids, token_embeds, ngram_range
    )
    assert len(seq_idx) == len(ngrams) == len(ngram_vecs)
    assert ngram_vecs.shape[1] == 384


def test_phrase_foundry_encode():
    """
    Test the `encode` method of the `PhraseFoundry` class.

    This test verifies that the `encode` method correctly processes a list of documents
    and returns the expected inputs and token embeddings.

    Steps:
    1. Initialize a `PhraseFoundry` instance with a specified model name.
    2. Define a list of documents to be encoded.
    3. Call the `encode` method with the documents, specifying `max_length`, `batch_size`, and `stride`.
    4. Assert that the returned inputs contain the key "input_ids".
    5. Assert that the shape of the token embeddings matches the number of documents.

    Raises:
        AssertionError: If the inputs do not contain "input_ids" or if the shape of the token embeddings is incorrect.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name)
    docs = ["This is a test document."]
    inputs, token_embeds = phrase_foundry.encode(
        docs, max_length=10, batch_size=1, stride=3
    )
    assert "input_ids" in inputs
    assert token_embeds.shape[0] == len(docs)


def test_phrase_foundry_encode_extract():
    """
    Test the `encode_extract` method of the `PhraseFoundry` class.

    This test verifies that the `encode_extract` method correctly processes a list of documents
    and returns the expected sequence indices, n-grams, and n-gram vectors.

    The test checks the following:
    - The lengths of the returned `seq_idx`, `ngrams`, and `ngram_vecs` are equal.
    - The second dimension of the `ngram_vecs` array is 384, which corresponds to the expected
        embedding size of the model.

    The `encode_extract` method is called with the following parameters:
    - `docs`: A list containing a single test document.
    - `max_length`: The maximum length of the sequences.
    - `batch_size`: The batch size for processing.
    - `ngram_range`: The range of n-grams to extract.
    - `stride`: The stride for the sliding window.

    Returns:
            None
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name)
    docs = ["This is a test document."]
    seq_idx, ngrams, ngram_vecs = phrase_foundry.encode_extract(
        docs, max_length=10, batch_size=1, ngram_range=(2, 3), stride=3
    )
    assert len(seq_idx) == len(ngrams) == len(ngram_vecs)
    assert ngram_vecs.shape[1] == 384


def test_phrase_foundry_encode_queries():
    """
    Test the `encode_queries` method of the `PhraseFoundry` class.

    This test verifies that the `encode_queries` method correctly encodes a list of queries
    into embeddings of the expected shape.

    Steps:
    1. Initialize a `PhraseFoundry` instance with a specified model name.
    2. Define a list of queries to be encoded.
    3. Encode the queries using the `encode_queries` method with specified parameters.
    4. Assert that the shape of the resulting embeddings matches the expected dimensions.

    Assertions:
    - The number of embeddings should match the number of queries.
    - The dimensionality of each embedding should be 384.

    Raises:
    - AssertionError: If the shape of the embeddings does not match the expected dimensions.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name)
    queries = ["test query"]
    query_embeds = phrase_foundry.encode_queries(
        queries, max_length=10, batch_size=1, stride=3
    )
    assert query_embeds.shape[0] == len(queries)
    assert query_embeds.shape[1] == 384


def test_encode_with_amp_enabled():
    """
    Test the `encode` method of the `PhraseFoundry` class with AMP enabled.

    This test verifies that the `encode` method correctly processes a list of documents
    and returns the expected inputs and token embeddings when AMP (Automatic Mixed Precision)
    is enabled.

    Steps:
    1. Initialize a `PhraseFoundry` instance with a specified model name.
    2. Define a list of documents to be encoded.
    3. Call the `encode` method with the documents, specifying `max_length`, `batch_size`, `stride`, and `amp=True`.
    4. Assert that the returned inputs contain the key "input_ids".
    5. Assert that the shape of the token embeddings matches the number of documents.

    Raises:
        AssertionError: If the inputs do not contain "input_ids" or if the shape of the token embeddings is incorrect.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name)
    docs = ["This is a test document."]
    inputs, token_embeds = phrase_foundry.encode(
        docs, max_length=10, batch_size=1, stride=3, amp=True
    )
    assert "input_ids" in inputs
    assert token_embeds.shape[0] == len(docs)


def test_encode_with_amp_disabled():
    """
    Test the `encode` method of the `PhraseFoundry` class with AMP disabled.

    This test verifies that the `encode` method correctly processes a list of documents
    and returns the expected inputs and token embeddings when AMP (Automatic Mixed Precision)
    is disabled.

    Steps:
    1. Initialize a `PhraseFoundry` instance with a specified model name.
    2. Define a list of documents to be encoded.
    3. Call the `encode` method with the documents, specifying `max_length`, `batch_size`, `stride`, and `amp=False`.
    4. Assert that the returned inputs contain the key "input_ids".
    5. Assert that the shape of the token embeddings matches the number of documents.

    Raises:
        AssertionError: If the inputs do not contain "input_ids" or if the shape of the token embeddings is incorrect.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name)
    docs = ["This is a test document."]
    inputs, token_embeds = phrase_foundry.encode(
        docs, max_length=10, batch_size=1, stride=3, amp=False
    )
    assert "input_ids" in inputs
    assert token_embeds.shape[0] == len(docs)


def test_encode_with_chunking():
    """
    Test the `encode` method of the `PhraseFoundry` class with chunking enabled.

    This test verifies that the `encode` method correctly processes a list of documents
    with chunking enabled and returns the expected inputs and token embeddings.

    Steps:
    1. Initialize a `PhraseFoundry` instance with a specified model name.
    2. Define a list of documents to be encoded.
    3. Call the `encode` method with the documents, specifying `max_length`, `batch_size`, `stride`, and `do_chunking=True`.
    4. Assert that the returned inputs contain the key "input_ids".
    5. Assert that the shape of the token embeddings matches the expected number of chunks.
    6. Verify that the last chunk tokens match the last tokens of the document.

    Raises:
        AssertionError: If the inputs do not contain "input_ids" or if the shape of the token embeddings is incorrect.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name)
    docs = [
        "This is a test document. " * 50
    ]  # Create a long document to ensure chunking
    max_length = 10
    stride = 5
    inputs, token_embeds = phrase_foundry.encode(
        docs, max_length=max_length, stride=stride, do_chunking=True
    )
    assert "input_ids" in inputs
    assert token_embeds.shape[0] == 99
    token_ids = phrase_foundry.tokenizer(docs, return_tensors="np")["input_ids"][0]
    # Check that the last chunk tokens match the last tokens of the document
    last_chunk_tokens = inputs["input_ids"][-1, 1:].cpu().numpy()
    last_chunk_tokens = last_chunk_tokens[
        last_chunk_tokens != phrase_foundry.tokenizer.pad_token_id
    ]
    np.testing.assert_equal(last_chunk_tokens, token_ids[-len(last_chunk_tokens) :])


def test_encode_without_chunking():
    """
    Test the `encode` method of the `PhraseFoundry` class with chunking disabled.

    This test verifies that the `encode` method correctly processes a list of documents
    without chunking and returns the expected inputs and token embeddings.

    Steps:
    1. Initialize a `PhraseFoundry` instance with a specified model name.
    2. Define a list of documents to be encoded.
    3. Call the `encode` method with the documents, specifying `max_length`, `batch_size`, `stride`, and `do_chunking=False`.
    4. Assert that the returned inputs contain the key "input_ids".
    5. Assert that the shape of the token embeddings matches the number of documents.

    Raises:
        AssertionError: If the inputs do not contain "input_ids" or if the shape of the token embeddings is incorrect.
    """
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    phrase_foundry = PhraseFoundry(model_name)
    docs = [
        "This is a test document. " * 50
    ]  # Create a long document to ensure chunking
    max_length = 10
    stride = 5
    inputs, token_embeds = phrase_foundry.encode(
        docs, max_length=max_length, stride=stride, do_chunking=False
    )
    assert "input_ids" in inputs
    assert token_embeds.shape[0] == len(docs)
