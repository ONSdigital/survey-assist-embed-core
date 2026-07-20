# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
