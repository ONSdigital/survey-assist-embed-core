"""Tests for the storage adapter and GCS retrieval asset helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from survey_assist_embed_core.adapters.storage.gcs import (
    DownloadedVectorStore,
    download_one_file_from_gcs,
    download_vector_store_from_gcs,
    is_gcs_path,
    parse_gcs_uri,
)
from survey_assist_embed_core.adapters.storage.local_gcs import LocalGcsStorage

NON_GCS_PATH = "local/path"


class _FakeBlob:
    """Fake GCS blob."""

    def __init__(self, exists: bool):
        self._exists = exists
        self.name = ""
        self.downloaded_to: str | None = None

    def exists(self):
        """Report whether the fake blob exists."""
        return self._exists

    def download_to_filename(self, filename: str):
        """Simulate downloading the blob by writing dummy content."""
        self.downloaded_to = filename
        Path(filename).write_text("test", encoding="utf-8")


class _FakeBucket:  # pylint: disable=too-few-public-methods
    """Fake GCS bucket."""

    def __init__(self, blob_map: dict[str, _FakeBlob]):
        self._blob_map = blob_map
        for name, blob in blob_map.items():
            blob.name = name

    def blob(self, name: str):
        """Return the fake blob for a blob name."""
        return self._blob_map[name]

    def list_blobs(self, prefix: str = ""):
        """Return fake blobs whose names match the prefix."""
        return [
            blob for name, blob in self._blob_map.items() if name.startswith(prefix)
        ]


class _FakeStorageClient:  # pylint: disable=too-few-public-methods
    """Fake GCS storage client."""

    def __init__(self, bucket_map: dict[str, _FakeBucket]):
        self._bucket_map = bucket_map

    def bucket(self, name: str):
        """Return the fake bucket for a bucket name."""
        return self._bucket_map[name]


@pytest.mark.utils
def test_is_gcs_path() -> None:
    """GCS URI detection should distinguish local and remote paths."""
    assert is_gcs_path("gs://bucket/path") is True
    assert is_gcs_path(NON_GCS_PATH) is False


@pytest.mark.utils
def test_parse_gcs_uri_valid() -> None:
    """A full GCS URI should split into bucket and prefix."""
    bucket, prefix = parse_gcs_uri("gs://my-bucket/path/to/store")

    assert bucket == "my-bucket"
    assert prefix == "path/to/store"


@pytest.mark.utils
def test_parse_gcs_uri_bucket_only() -> None:
    """A bucket-only URI should produce an empty prefix."""
    bucket, prefix = parse_gcs_uri("gs://my-bucket")

    assert bucket == "my-bucket"
    assert prefix == ""


@pytest.mark.utils
def test_parse_gcs_uri_invalid() -> None:
    """Non-GCS paths should be rejected."""
    with pytest.raises(ValueError, match="Not a valid GCS URI"):
        parse_gcs_uri("local/not-gcs")


@pytest.mark.utils
def test_parse_gcs_uri_missing_bucket() -> None:
    """A GCS URI with no bucket should be rejected."""
    with pytest.raises(ValueError, match="bucket missing"):
        parse_gcs_uri("gs://")


@pytest.mark.utils
def test_download_vector_store_from_gcs_success() -> None:
    """The helper should download all files under the requested prefix."""
    metadata_blob = _FakeBlob(exists=True)
    vectors_blob = _FakeBlob(exists=True)
    extra_blob = _FakeBlob(exists=True)

    fake_client = _FakeStorageClient(
        {
            "my-bucket": _FakeBucket(
                {
                    "prefix/metadata.json": metadata_blob,
                    "prefix/vectors.parquet": vectors_blob,
                    "prefix/extra/readme.txt": extra_blob,
                }
            )
        }
    )

    with patch(
        "survey_assist_embed_core.adapters.storage.gcs.Client",
        return_value=fake_client,
    ):
        downloaded = download_vector_store_from_gcs("gs://my-bucket/prefix")

    assert isinstance(downloaded, DownloadedVectorStore)
    assert Path(downloaded.path, "metadata.json").exists()
    assert Path(downloaded.path, "vectors.parquet").exists()
    assert Path(downloaded.path, "extra", "readme.txt").exists()
    assert downloaded.temp_dir is not None

    downloaded.cleanup()


@pytest.mark.utils
def test_download_vector_store_from_gcs_bucket_root_success() -> None:
    """A bucket-root URI should download files without a prefix."""
    metadata_blob = _FakeBlob(exists=True)
    vectors_blob = _FakeBlob(exists=True)

    fake_client = _FakeStorageClient(
        {
            "my-bucket": _FakeBucket(
                {
                    "metadata.json": metadata_blob,
                    "vectors.parquet": vectors_blob,
                }
            )
        }
    )

    with patch(
        "survey_assist_embed_core.adapters.storage.gcs.Client",
        return_value=fake_client,
    ):
        downloaded = download_vector_store_from_gcs("gs://my-bucket")

    assert isinstance(downloaded, DownloadedVectorStore)
    assert Path(downloaded.path, "metadata.json").exists()
    assert Path(downloaded.path, "vectors.parquet").exists()

    downloaded.cleanup()


@pytest.mark.utils
def test_download_vector_store_from_gcs_missing_prefix() -> None:
    """The helper should report when no files exist under the store path."""
    other_blob = _FakeBlob(exists=True)

    fake_client = _FakeStorageClient(
        {
            "my-bucket": _FakeBucket(
                {
                    "other-prefix/metadata.json": other_blob,
                }
            )
        }
    )

    with (
        patch(
            "survey_assist_embed_core.adapters.storage.gcs.Client",
            return_value=fake_client,
        ),
        pytest.raises(FileNotFoundError, match=r"No files found in GCS store path"),
    ):
        download_vector_store_from_gcs("gs://my-bucket/prefix")


@pytest.mark.utils
def test_download_vector_store_from_gcs_directory_markers_only() -> None:
    """Directory marker blobs should be ignored and still count as no files."""
    directory_blob = _FakeBlob(exists=True)

    fake_client = _FakeStorageClient(
        {
            "my-bucket": _FakeBucket(
                {
                    "prefix/": directory_blob,
                }
            )
        }
    )

    with (
        patch(
            "survey_assist_embed_core.adapters.storage.gcs.Client",
            return_value=fake_client,
        ),
        pytest.raises(FileNotFoundError, match=r"No files found in GCS store path"),
    ):
        download_vector_store_from_gcs("gs://my-bucket/prefix")

    assert directory_blob.downloaded_to is None


@pytest.mark.utils
def test_download_one_file_from_gcs_success() -> None:
    """The helper should download one remote source file into temp storage."""
    file_blob = _FakeBlob(exists=True)

    fake_client = _FakeStorageClient(
        {
            "my-bucket": _FakeBucket(
                {
                    "path/to/data.csv": file_blob,
                }
            )
        }
    )

    with patch(
        "survey_assist_embed_core.adapters.storage.gcs.Client",
        return_value=fake_client,
    ):
        downloaded = download_one_file_from_gcs("gs://my-bucket/path/to/data.csv")

    assert isinstance(downloaded, DownloadedVectorStore)
    assert Path(downloaded.path).name == "data.csv"
    assert Path(downloaded.path).exists()
    assert downloaded.temp_dir is not None

    downloaded.cleanup()


@pytest.mark.utils
def test_downloaded_vector_store_context_manager_cleans_up_file() -> None:
    """The returned wrapper should support scoped cleanup with ``with``."""
    temp_path: Path | None = None

    with DownloadedVectorStore(
        path=str(Path("placeholder")),
        temp_dir=tempfile.TemporaryDirectory(),
    ) as downloaded:
        temp_path = Path(downloaded.temp_dir.name)
        downloaded_file = temp_path / "payload.txt"
        downloaded_file.write_text("test", encoding="utf-8")

        assert temp_path.exists()
        assert downloaded_file.exists()

    assert temp_path is not None
    assert not temp_path.exists()


@pytest.mark.utils
def test_download_one_file_from_gcs_missing_file() -> None:
    """The helper should report a missing source file clearly."""
    file_blob = _FakeBlob(exists=False)

    fake_client = _FakeStorageClient(
        {
            "my-bucket": _FakeBucket(
                {
                    "path/to/data.csv": file_blob,
                }
            )
        }
    )

    with (
        patch(
            "survey_assist_embed_core.adapters.storage.gcs.Client",
            return_value=fake_client,
        ),
        pytest.raises(
            FileNotFoundError,
            match=r"gs://my-bucket/path/to/data\.csv",
        ),
    ):
        download_one_file_from_gcs("gs://my-bucket/path/to/data.csv")


@pytest.mark.utils
def test_local_gcs_storage_returns_local_paths() -> None:
    """Local paths should pass through unchanged."""
    storage = LocalGcsStorage()

    assert storage.resolve_store_path(path=NON_GCS_PATH) == NON_GCS_PATH
    assert storage.resolve_source_file(path=NON_GCS_PATH) == NON_GCS_PATH


@pytest.mark.utils
def test_local_gcs_storage_downloads_remote_store(tmp_path: Path) -> None:
    """A remote store path should be downloaded and retained by the adapter."""
    downloaded = DownloadedVectorStore(
        path=str(tmp_path / "downloaded-store"),
        temp_dir=SimpleNamespace(name=str(tmp_path / "downloaded-store")),
    )
    storage = LocalGcsStorage()

    with patch(
        "survey_assist_embed_core.adapters.storage.local_gcs.download_vector_store_from_gcs",
        return_value=downloaded,
    ) as mock_download:
        resolved_path = storage.resolve_store_path(path="gs://my-bucket/prefix")

    assert resolved_path == downloaded.path
    assert storage._downloads == [downloaded]  # pylint: disable=protected-access
    mock_download.assert_called_once_with("gs://my-bucket/prefix")


@pytest.mark.utils
def test_local_gcs_storage_downloads_remote_source_file(tmp_path: Path) -> None:
    """A remote source file should be downloaded and retained by the adapter."""
    downloaded = DownloadedVectorStore(
        path=str(tmp_path / "data.csv"),
        temp_dir=SimpleNamespace(name=str(tmp_path)),
    )
    storage = LocalGcsStorage()

    with patch(
        "survey_assist_embed_core.adapters.storage.local_gcs.download_one_file_from_gcs",
        return_value=downloaded,
    ) as mock_download:
        resolved_path = storage.resolve_source_file(path="gs://my-bucket/data.csv")

    assert resolved_path == downloaded.path
    assert storage._downloads == [downloaded]  # pylint: disable=protected-access
    mock_download.assert_called_once_with("gs://my-bucket/data.csv")
