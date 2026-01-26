# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-01-26

### Added
- `max_chunk_tokens` parameter for token-based chunking limits
- `split_long_sents` parameter to control sentence splitting behavior at token boundaries
- `idx` column to output DataFrames for direct embedding indexing (starts at 0)
- `max_chunk_sents` and `max_chunk_tokens` columns to DataFrame output for configuration tracking
- Aligned lists feature for multi-configuration experiments (pairing `max_chunk_sents` and `max_chunk_tokens`)
- GitHub CI workflow for automated testing
- Improved type annotations and fixed Polars dtype issues

### Changed
- **BREAKING**: Renamed `Encoder` class to `LateEncoder` (deprecated alias provided for backwards compatibility)
- **BREAKING**: Renamed `num_sents` parameter to `max_chunk_sents` for consistency
- **BREAKING**: Renamed `chunk_overlap` parameter to `chunk_overlap_sents` for clarity
- **BREAKING**: Renamed `prechunk_overlap` parameter to `prechunk_overlap_tokens` for consistency
- Updated deduplication to include configuration parameters in compound key
- Improved warning messages to include sequence indices for better debugging

### Fixed
- DataFrame idx column now starts at 0 (previously started at 1 after sorting)
- Validation now raises `ValueError` if `max_chunk_tokens` exceeds model's `max_length`
- Deduplication now works correctly for split chunks from long sentences

## [0.1.0] - 2026-01-17

### Added
- Initial release of afterthoughts
- Late chunking implementation for context-aware sentence-chunk embeddings
- Support for multiple sentence tokenizers (BlingFire, NLTK, pysbd, syntok)
- Dynamic batching by token count
- Memory optimizations (float16 conversion, dimension truncation)
- Polars and pandas DataFrame output support
- Comprehensive test suite
- Documentation and examples

[Unreleased]: https://github.com/ndgigliotti/afterthoughts/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/ndgigliotti/afterthoughts/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ndgigliotti/afterthoughts/releases/tag/v0.1.0
