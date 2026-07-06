# Survey Assist Embed Core

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/github/license/ONSdigital/survey-assist-embed-core)
![Release](https://img.shields.io/github/v/release/ONSdigital/survey-assist-embed-core)
![Status](https://img.shields.io/badge/code%20status-in%20development%20-red)

## Overview

Common embedding logic functionality used in Survey Assist vector stores.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Running Locally](#running-locally)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Security](#security)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Additional Documentation](#additional-documentation)

## Features

- Build persisted vector-store artifacts from labelled CSV data.
- Load existing local or GCS-backed vector stores through `EmbeddingHandler`.
- Run single-query and multi-field retrieval against a loaded index.
- Inspect runtime backend and index configuration through typed status models.

## Architecture

The package is structured around an explicit two-step workflow:

1. Build persisted retrieval artifacts with `build_embedding_index`.
2. Load those artifacts with `EmbeddingHandler` and perform searches.

At runtime, `EmbeddingHandler` provides the public API for retrieval while the
ClassifAI adapter handles vector-store creation and loading. Storage helpers
resolve local filesystem paths and GCS-backed inputs.

## Prerequisites

- Python 3.12
- Poetry 2.1.3 or later
- Git
- Make
- Docker / Colima (optional) / Podman
- Google Cloud SDK

## Local Development Setup

### Clone Repository

```shell
git clone https://github.com/ONSdigital/survey-assist-embed-core.git
```

### Install Dependencies

```shell
poetry install
```

### Install Git Hooks

This repository uses pre-commit hooks to perform code quality,
security, and secret-scanning checks before code is committed.

```shell
poetry run pre-commit install
poetry run pre-commit install --hook-type pre-push
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## Running Locally

The quickest way to exercise it locally is to run the demo script or use the
package directly from a Python session.

### Run the Demo

```shell
poetry run python demos/embedding_handler.py
```

The demo script creates a small temporary `label,text` CSV, builds persisted
vector-store artifacts, loads them with `EmbeddingHandler`, and prints example
search results.

If you prefer interactive execution in VS Code, open
`demos/embedding_handler.py` and run the `# %%` cells one at a time.

### Use the Library Directly

```python
from survey_assist_embed_core import EmbeddingHandler, build_embedding_index

build_embedding_index(
    index_source_file="my_knowledgebase.csv",
    output_dir="vector_store",
)

handler = EmbeddingHandler(db_dir="vector_store")
results = handler.search_index("primary education")
```

## Configuration

The main entry points are:

- `build_embedding_index(...)`: builds persisted vector-store artifacts from a
  local CSV file or GCS URI.
- `EmbeddingHandler(...)`: loads a persisted vector store from local storage or
  GCS and exposes the retrieval methods.

Key inputs:

- `index_source_file`: CSV input containing `label,text` columns.
- `output_dir`: directory where persisted artifacts such as `metadata.json`
  and `vectors.parquet` are written.
- `db_dir`: local path or `gs://...` URI for a persisted vector store.
- `k_matches`: maximum number of results returned per search.
- `embedding_model_name`: optional model override for the build step. The
  default is `sentence-transformers/all-MiniLM-L6-v2`.

For GCS-backed builds or loads, ensure Google Cloud credentials are available
in the local environment before running the workflow.

## Repository Structure

An end-to-end cell-based demo is available at `demos/embedding_handler.py`.

```txt
survey-assist-embed-core/
|-- demos/                          # cell-based usage demos (# %%)
|-- docs/                           # MkDocs documentation
|-- src/survey_assist_embed_core/   # main library package
|   |-- adapters/                   # ClassifAI and storage adapters
|   |-- embed/                      # EmbeddingHandler public retrieval API
|   |-- models/                     # pydantic config and response models
|   |-- ports/                      # backend protocols
|-- tests/                          # pytest unit tests
|-- .github/                        # GitHub actions workflows
|-- README.md                       # This file
```

## Testing

Run all of the unit tests

```shell
make all-tests
```

Additional testing guidance is available in [CONTRIBUTING.md](CONTRIBUTING.md).

## Code Quality

Running validation checks:

```shell
make check-python-nofix
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and validation requirements.

## Security

Security checks are performed using:

- Bandit
- GitHub Dependabot
- Secret scanning

Security vulnerabilities **must not** be disclosed publicly.

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Documentation

Documentation is maintained using MkDocs.

```shell
make run-docs
```

## Contributing

Please read [the contribution guidelines](CONTRIBUTING.md) before creating a pull request.

## Additional Documentation

[CONTRIBUTING.md](CONTRIBUTING.md)

[RELEASING.md](RELEASING.md)

[SECURITY.md](SECURITY.md)

[CHANGELOG.md](CHANGELOG.md)

[LICENSE.md](LICENSE.md)

[CODEOWNERS](CODEOWNERS)
