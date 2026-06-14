"""Smoke test: index a PDF and run a few queries against it.

Usage (from the project root, with the venv active and services running):

    python scripts/smoketest.py

Prerequisites:
    - Elasticsearch and MinIO must be reachable (docker compose up)
    - A .env file with MINIO_ROOT_USER, MINIO_ROOT_PASSWORD, JWT_SECRET_KEY,
    and at least one of OPENAI_API_KEY / HF_API_TOKEN set
    - data/faqs/faq-1.pdf must exist
"""

import sys
import textwrap
from pathlib import Path

import needle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDF_PATH = Path("data/faqs/faq-1.pdf")
BUSINESS_ID = "smoketest"
TEST_INDEX = "smoketest"

QUERIES = [
    "When is lunch time?",
    "When will we get paid?",
    "How do I apply for leave?",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP_LENGTH = 72
SEP_CHAR = "─"


def print_section(title: str) -> None:
    line = f"{f' {title} ':{SEP_CHAR}^{SEP_LENGTH}}"
    print(f"\n{line}")


def check(condition: bool, message: str) -> None:
    icon = "✓" if condition else "✗"
    print(f"  {icon}  {message}")
    if not condition:
        sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Connect
# ---------------------------------------------------------------------------

print_section("Connecting to Services")

statuses = needle.connect(es_index=TEST_INDEX)
print(f"  Service statuses: {statuses}")
print(f"  Using test index: {TEST_INDEX}")

check(statuses.get("elasticsearch"), "Elasticsearch is reachable")
check(statuses.get("s3"), "S3 is reachable")

# ---------------------------------------------------------------------------
# 2. Index
# ---------------------------------------------------------------------------

print_section("Indexing")

check(PDF_PATH.exists(), f"Test PDF exists at {PDF_PATH}")

content = PDF_PATH.read_bytes()
print(f"  File size: {len(content):,} bytes")

storage_path = needle.index(
    business_id=BUSINESS_ID,
    content=content,
    mime_type="application/pdf",
    original_path=PDF_PATH.name,
)
print(f"  Indexed as: {storage_path}")

check(bool(storage_path), "index() returned a non-empty storage path")

# ---------------------------------------------------------------------------
# 3. List
# ---------------------------------------------------------------------------

print_section("Listing Files")

files = needle.list_(business_id=BUSINESS_ID)

print(f"  Files in bucket '{BUSINESS_ID}': {len(files)}")
for f in files:
    print(f"    • {f['key']}  ({f.get('content_type', '?')},  {f.get('size', '?')} bytes)")

check(any(f["key"] == storage_path for f in files), "Indexed file appears in listing")

# ---------------------------------------------------------------------------
# 4a. Query — full pipeline (generate_response=True)
# ---------------------------------------------------------------------------

print_section("Querying (with LLM response)")

for i, query in enumerate(QUERIES, 1):
    print(f"\n  [{i}] {query}")

    result = needle.query(business_id=BUSINESS_ID, text=query, generate_response=True)
    answer = result.get("answer", "").strip()
    chunks = result.get("source_chunks", [])

    print(f"  Answer:\n{textwrap.indent(answer, '    ')}")
    print(f"  Sources: {len(chunks)} chunk(s) retrieved")
    for chunk in chunks:
        score = chunk.get("score")
        path = chunk.get("storage_path", "?")
        preview = (chunk.get("content") or "")[:SEP_LENGTH].replace("\n", " ")
        print(f"    • [{score:.3f}] {path}: {preview!r}")

    check(bool(answer), f"Query {i} returned a non-empty answer")
    check(len(chunks) > 0, f"Query {i} returned at least one source chunk")

# ---------------------------------------------------------------------------
# 4b. Query — retrieve only (generate_response=False)
# ---------------------------------------------------------------------------

print_section("Querying (no LLM response)")

sample_query = QUERIES[0]
print(f"\n  Query: {sample_query}")

result = needle.query(business_id=BUSINESS_ID, text=sample_query, generate_response=False)
answer = result.get("answer")
chunks = result.get("source_chunks", [])

print(f"  Answer: {answer!r}  (expected None)")
print(f"  Sources: {len(chunks)} chunk(s) retrieved")
for chunk in chunks:
    score = chunk.get("score")
    path = chunk.get("storage_path", "?")
    preview = (chunk.get("content") or "")[:SEP_LENGTH].replace("\n", " ")
    print(f"    • [{score:.3f}] {path}: {preview!r}")

check(answer is None, "retrieve-only query returned answer=None")
check(len(chunks) > 0, "retrieve-only query returned at least one source chunk")

# ---------------------------------------------------------------------------
# 5. Clean up
# ---------------------------------------------------------------------------

print_section("Cleaning Up")

result = needle.delete(business_id=BUSINESS_ID, storage_path=storage_path)
print(f"  {result['message']}")
check("deleted" in result["message"].lower(), "File deleted from storage and index")

files_after = needle.list_(business_id=BUSINESS_ID)
check(
    all(f["key"] != storage_path for f in files_after),
    "File no longer appears in listing after deletion",
)

print(f"\n  Cleaning up test index: {TEST_INDEX}")
try:
    from needle.storage import get_document_store

    ds = get_document_store()
    if ds._client is not None:
        ds._client.indices.delete(index=TEST_INDEX, ignore_unavailable=True)

    print(f"  ✓ Test index '{TEST_INDEX}' deleted")
except Exception as exc:
    print(f"  ⚠ Warning: Could not delete test index: {exc}")

print(f"\n  Cleaning up test bucket: {BUSINESS_ID}")
try:
    from needle.storage import get_s3_storage

    s3 = get_s3_storage()
    s3.client.delete_bucket(Bucket=BUSINESS_ID)

    print(f"  ✓ Test bucket '{BUSINESS_ID}' deleted")
except Exception as exc:
    print(f"  ⚠ Warning: Could not clean up test bucket: {exc}")

# ---------------------------------------------------------------------------

print_section("✓ All Checks Passed")
print()
