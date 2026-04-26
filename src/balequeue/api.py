import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status

from .config import settings
from .pipelines import IndexingPipeline, QueryPipeline
from .schemas import QueryRequest, QueryResponse, SourceChunk, UploadResponse

router = APIRouter()

SUPPORTED_MIME_TYPES = {"text/plain"}

# TODO: Are pipelines thread-safe?
# If not, we may need to create new instances per request or use locks.
indexing_pipeline = IndexingPipeline()
query_pipeline = QueryPipeline()


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/upload/{business_id}",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    business_id: str,
    file: UploadFile,
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename.",
        )

    if file.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Accepted: {SUPPORTED_MIME_TYPES}",
        )

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix
    ) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        _ = indexing_pipeline.run(
            data={
                "converter": {
                    "sources": [temp_path],
                    "meta": {
                        "business_id": business_id,
                    },
                },
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {exc}",
        ) from exc
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return UploadResponse(
        message="Document uploaded successfully.", business_id=business_id
    )


@router.post("/query", response_model=QueryResponse)
async def query_business(request: QueryRequest) -> QueryResponse:
    try:
        result = query_pipeline.run(
            data={
                "embedder": {"text": request.query},
                "retriever": {
                    "filters": {
                        "field": "meta.business_id",
                        "operator": "==",
                        "value": request.business_id,  # TODO: We should use business_id in tandem with a secret key to prevent unauthorized access to data from other businesses
                    },
                    "top_k": request.top_k or settings.top_k,
                },
                "prompt_builder": {"query": request.query},
            },
            include_outputs_from=["generator", "retriever"],
        )

        answer = result["generator"]["replies"][0]
        documents = result["retriever"]["documents"]

        source_chunks = [
            SourceChunk(
                content=doc.content,
                score=doc.score,
                file_path=doc.meta["file_path"],
            )
            for doc in documents
        ]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {exc}",
        ) from exc

    return QueryResponse(
        answer=answer,
        source_chunks=source_chunks,
    )
