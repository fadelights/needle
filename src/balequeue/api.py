import tempfile
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordRequestForm
from haystack.dataclasses import ByteStream
from sqlalchemy.orm import Session

from .auth import (
    create_access_token,
    get_current_business,
    get_db,
    get_passwd_hash,
    verify_passwd,
)
from .config import settings
from .models import Business
from .pipelines import IndexingPipeline, QueryPipeline
from .schemas import (
    BusinessCreate,
    BusinessOut,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    Token,
    UploadResponse,
)
from .storage import s3_storage

router = APIRouter()

SUPPORTED_MIME_TYPES = {
    "application/json",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # XLSX
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "text/csv",
    "text/html",
    "text/markdown",
    "text/plain",
}

# TODO: Are pipelines thread-safe?
# If not, we may need to create new instances per request or use locks.
indexing_pipeline = IndexingPipeline()
query_pipeline = QueryPipeline()


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/auth/register", response_model=BusinessOut, status_code=status.HTTP_201_CREATED
)
async def register_business(
    business_in: BusinessCreate, db: Session = Depends(get_db)
) -> BusinessOut:
    db_business = db.query(Business).filter(Business.name == business_in.name).first()
    if db_business:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Business name already registered",
        )
    hashed_password = get_passwd_hash(business_in.password)
    new_business = Business(name=business_in.name, hashed_password=hashed_password)
    db.add(new_business)
    db.commit()
    db.refresh(new_business)
    return new_business


@router.post("/auth/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
) -> Token:
    business = db.query(Business).filter(Business.name == form_data.username).first()
    if not business or not verify_passwd(form_data.password, business.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": business.name}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    file: UploadFile,
    current_business: Business = Depends(get_current_business),
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

    file_content = await file.read()
    try:
        s3_storage.upload_file(
            bucket=current_business.business_id,
            obj=file.filename,
            data=file_content,
            content_type=file.content_type,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document to object storage: {exc}",
        ) from exc

    byte_stream = ByteStream(
        data=file_content,
        mime_type=file.content_type,
    )

    try:
        _ = indexing_pipeline.run(
            data={
                "converter": {
                    "sources": [
                        byte_stream,
                    ],
                    "meta": {
                        "business_id": current_business.business_id,
                        "file_path": file.filename,
                    },
                },
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {exc}",
        ) from exc

    return UploadResponse(
        message="Document uploaded successfully.",
        business_id=current_business.business_id,
    )


@router.post("/query", response_model=QueryResponse)
async def query_business(
    request: QueryRequest,
    current_business: Business = Depends(get_current_business),
) -> QueryResponse:
    try:
        result = query_pipeline.run(
            data={
                "embedder": {"text": request.query},
                "retriever": {
                    "filters": {
                        "field": "meta.business_id",
                        "operator": "==",
                        "value": current_business.business_id,
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
