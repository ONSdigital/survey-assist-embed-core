# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-17

### Added

- Added support for query-length-dependent retriever weights through dedicated WeightSpecs class.
- Added automatic weight normalisation across retrievers, including mixed fixed and query-length-dependent weight configurations.
- Strict suggestion limits are now applied by default in SAYT suggest functions and SAYTSuggester through the `include_tied_suggestions` parameter.

### Changed

- Refactored SAYT weighting to use the dedicated WeightSpecs class instead of defining weights within retriever specifications, now enforcing a single retriever instance per retriever type.
- Updated the default SAYT retriever weights to n4_p5_s1 for 3–5 characters and n3_p4_s3 for 6+ characters.
- Updated the default SAYT `min_chars` to 3 to align with new weight defaults.
- Updated Vulnerable dependencies

### Fixed

- Passed the configured device to fastembed's ONNX embedding model while retaining
  automatic device selection when unspecified.
- Updated SAYT suggestions to exclude non-positive scores from DenseVectorIndex, preventing zero or negative-scored suggestions from being returned.

## [0.2.3] - 2026-07-20

### Fixed

- Updated vector-store loading to resolve the persisted ClassifAI
  `vectoriser_class` value directly, including concrete class-name spellings.

## [0.2.2] - 2026-07-17

### Fixed

- Fixed SAYT artifact building so documented `gs://` source CSV URIs work.
- Reused the shared GCS-to-local file resolution helper across both vector-store
  and SAYT artifact-loading paths.

## [0.2.1] - 2026-07-17

### Added

- Added an ONNX vectoriser option for semantic embedding.

### Changed

- Upgraded ClassifAI to v1.1.1.
- Aligned SAYT integration with the updated ClassifAI dependency.

## [0.2.0] - 2026-07-08

### Added

- Migrated embedding functionality from sic-classification-utils into this package.
- Added structured logging support from survey-assist-utils.
- Migrated search-as-you-type (SAYT) functionality from sic-classification-utils.

### Changed

- Added artefact persistence logic to the VectorBackend contract.
- Updated docstrings and demos.
- Applied SAYT tweaks needed for the 100examples evaluation.

## [0.1.0] - 2026-06-17

### Added

- Base files for README, CHANGELOG, LICENSE, CODEOWNERS and CONTRIBUTING.
- Initial project structure.
- GitHub Actions workflow for linting, Bandit security checks and unit tests.

## Placeholder

This section is kept to have easy access to the types change to capture.

### Added

### Changed

### Deprecated

### Fixed

### Security
