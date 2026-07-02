"""End-to-end EmbeddingHandler demo.

This demo uses the new two-step workflow in ``survey-assist-embed-core``:

1. Build persisted vector-store artifacts with ``build_embedding_index``.
2. Load those artifacts with ``EmbeddingHandler`` and run searches.
"""

# %%
import tempfile
from pathlib import Path

from survey_assist_embed_core import EmbeddingHandler, build_embedding_index

# %%
# Create a tiny self-contained CSV so the demo does not depend on external
# package data or checked-in vector-store artifacts.
workspace = Path(tempfile.mkdtemp(prefix="embed-core-demo-"))
source_csv = workspace / "toy_index.csv"
vector_store_dir = workspace / "vector_store"

source_csv.write_text(
    "label,text\n"
    "47190,Retail sale in non-specialised stores\n"
    "62012,Business and domestic software development\n"
    "56101,Licensed restaurants\n"
    "85200,Primary education\n",
    encoding="utf-8",
)

print(f"Demo workspace: {workspace}")
print(f"Source CSV: {source_csv}")


# %%
# Build the persisted vector-store artifacts. The first run may download the
# default sentence-transformer model if it is not already cached locally.
print("Building vector-store artifacts...")
build_embedding_index(
    index_source_file=str(source_csv),
    output_dir=str(vector_store_dir),
)
print("Built artifacts:", sorted(path.name for path in vector_store_dir.iterdir()))


# %%
# Load the handler from the saved artifacts. EmbeddingHandler is load-only, so
# it expects a pre-built vector store.
handler = EmbeddingHandler(db_dir=str(vector_store_dir))
print(handler.get_embed_config().model_dump_json(indent=2))


# %%
# Run a single-query search.
single_query = "retail assistant"
single_results = handler.search_index(single_query)
print(f"Results for {single_query!r}:")
print(single_results.model_dump_json(indent=2))


# %%
# Run a multi-field search. None values are ignored, which is useful when the
# upstream caller has optional fields.
multi_results = handler.search_index_multi(["primary", None, "education"])
print("Results for multi-field query ['primary', None, 'education']:")
print(multi_results.model_dump_json(indent=2))


# %%
# If you already have a persisted vector store on disk or in GCS, you can skip
# the build step entirely and load it directly.
#
# handler = EmbeddingHandler(db_dir="vector_store")
# handler = EmbeddingHandler(db_dir="gs://my-bucket/vector_store")
