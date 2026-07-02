"""GCS helpers for retrieval asset access."""

import os
import tempfile
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from google.cloud.storage import Client
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
from survey_assist_utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DownloadedVectorStore:
    """Represent a vector store or source file downloaded into a temp directory."""

    path: str
    temp_dir: tempfile.TemporaryDirectory[str]

    def cleanup(self) -> None:
        """Delete the temporary directory backing the downloaded asset."""
        self.temp_dir.cleanup()

    def __enter__(self) -> Self:
        """Return the downloaded asset for scoped use."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Clean up the temporary directory when leaving a ``with`` block."""
        del exc_type, exc, traceback
        self.cleanup()


def is_gcs_path(path: str) -> bool:
    """Return whether a path is a ``gs://`` URI."""
    return path.startswith("gs://")


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse ``gs://bucket/path`` into ``(bucket_name, path)``."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Not a valid GCS URI: {gcs_uri}")

    without_scheme = gcs_uri[5:]
    parts = without_scheme.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

    if not bucket_name:
        raise ValueError(f"Invalid GCS URI, bucket missing: {gcs_uri}")

    return bucket_name, prefix


def download_vector_store_from_gcs(gcs_uri: str) -> DownloadedVectorStore:
    """Download all files under a GCS store path into a temp directory."""
    bucket_name, prefix = parse_gcs_uri(gcs_uri)
    blob_prefix = f"{prefix}/" if prefix else ""

    client = Client()
    bucket: Bucket = client.bucket(bucket_name)
    blobs: list[Blob] = list(bucket.list_blobs(prefix=blob_prefix))

    if not blobs:
        raise FileNotFoundError(f"No files found in GCS store path: {gcs_uri}")

    temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    local_dir = temp_dir.name
    downloaded_files = 0

    for blob in blobs:
        relative_name = blob.name[len(blob_prefix) :] if blob_prefix else blob.name
        if not relative_name or relative_name.endswith("/"):
            continue

        local_path = os.path.join(local_dir, relative_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        downloaded_files += 1

    if downloaded_files == 0:
        temp_dir.cleanup()
        raise FileNotFoundError(f"No files found in GCS store path: {gcs_uri}")

    logger.info(
        "Downloaded vector store from GCS to local directory.",
        uri=gcs_uri,
        local_dir=local_dir,
    )

    return DownloadedVectorStore(path=local_dir, temp_dir=temp_dir)


def download_one_file_from_gcs(gcs_uri: str) -> DownloadedVectorStore:
    """Download one file from GCS into a temporary directory."""
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)

    client = Client()
    bucket: Bucket = client.bucket(bucket_name)
    blob: Blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(f"File not found in GCS: {gcs_uri}")

    temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    local_path = os.path.join(temp_dir.name, os.path.basename(blob_name))
    blob.download_to_filename(local_path)

    logger.info(
        "Downloaded file from GCS to local path.",
        uri=gcs_uri,
        local_path=local_path,
    )

    return DownloadedVectorStore(path=local_path, temp_dir=temp_dir)
