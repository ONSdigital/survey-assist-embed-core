"""Storage adapter that passes through local paths and downloads GCS URIs."""

from survey_assist_embed_core.adapters.storage.gcs import (
    DownloadedVectorStore,
    download_one_file_from_gcs,
    download_vector_store_from_gcs,
    is_gcs_path,
)
from survey_assist_embed_core.ports import Storage


class LocalGcsStorage(Storage):
    """Resolve local paths directly and download GCS paths into temp storage."""

    def __init__(self) -> None:
        """Keep downloaded assets alive for as long as the adapter is retained."""
        self._downloads: list[DownloadedVectorStore] = []

    def _remember(self, downloaded: DownloadedVectorStore) -> str:
        """Retain a downloaded asset and return the local path to use."""
        self._downloads.append(downloaded)
        return downloaded.path

    def resolve_store_path(self, *, path: str) -> str:
        """Return a local folder path for a store location."""
        if not is_gcs_path(path):
            return path

        downloaded = download_vector_store_from_gcs(path)
        return self._remember(downloaded)

    def resolve_source_file(self, *, path: str) -> str:
        """Return a local file path for a source-file location."""
        if not is_gcs_path(path):
            return path

        downloaded = download_one_file_from_gcs(path)
        return self._remember(downloaded)
