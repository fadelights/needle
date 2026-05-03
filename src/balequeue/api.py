import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

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
    FileContentResponse,
    FileUpdateRequest,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    Token,
    UploadResponse,
)
from .storage import document_store, s3_storage

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


async def _process_upload(
    business_id: str, file_path: str, content: bytes, mime_type: str
):
    """
    Helper function to upload file to S3 and run the indexing pipeline.
    Used by both upload and update endpoints.
    """
    try:
        s3_storage.upload_file(
            bucket=business_id,
            obj=file_path,
            data=content,
            content_type=mime_type,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document to object storage: {exc}",
        ) from exc

    byte_stream = ByteStream(data=content, mime_type=mime_type)
    try:
        indexing_pipeline.run(
            data={
                "converter": {
                    "sources": [byte_stream],
                    "meta": {
                        "business_id": business_id,
                        "file_path": file_path,
                    },
                },
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {exc}",
        ) from exc


async def _process_deletion(business_id: str, file_path: str) -> Dict[str, str]:
    """
    Helper function to delete file from S3 and document store.
    Used by both delete and update endpoints.
    """
    try:
        s3_storage.delete_file(bucket=business_id, obj=file_path)
    except FileNotFoundError:
        # TODO: If the file is already missing from storage, we should still attempt to delete any documents that reference it.
        return {"message": f"Document '{file_path}' does not exist."}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document from object storage: {exc}",
        ) from exc

    filters = {
        "operator": "AND",
        "conditions": [
            {
                "field": "meta.business_id",
                "operator": "==",
                "value": business_id,
            },
            {
                "field": "meta.file_path",
                "operator": "==",
                "value": file_path,
            },
        ],
    }

    try:
        documents = document_store.filter_documents(filters=filters)
        ids = [document.id for document in documents]
        if ids:
            document_store.delete_documents(document_ids=ids)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document from document store: {exc}",
        ) from exc

    return {"message": f"Document '{file_path}' deleted successfully."}


# TODO: Handle duplicate uploads
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
    await _process_upload(
        business_id=current_business.business_id,
        file_path=file.filename,
        content=file_content,
        mime_type=file.content_type,
    )

    return UploadResponse(
        message="Document uploaded successfully.",
        business_id=current_business.business_id,
    )


@router.get("/list", response_model=List[str])
async def list_files(
    current_business: Business = Depends(get_current_business),
) -> List[str]:
    try:
        return s3_storage.list_files(bucket=current_business.business_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {exc}",
        ) from exc


@router.delete("/delete")
async def delete_file(
    file_path: str,
    current_business: Business = Depends(get_current_business),
) -> Dict[str, str]:
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File name must be provided.",
        )

    message = await _process_deletion(
        business_id=current_business.business_id,
        file_path=file_path,
    )

    return message


@router.get("/files/{file_path:path}", response_model=FileContentResponse)
async def get_file_content(
    file_path: str,
    current_business: Business = Depends(get_current_business),
) -> FileContentResponse:
    try:
        content_bytes = s3_storage.get_file(
            bucket=current_business.business_id, obj=file_path
        )
        # We assume UTF-8 for editable files (txt, md)
        content = content_bytes.decode("utf-8")
        return FileContentResponse(content=content, file_path=file_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_path}' not found.",
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{file_path}' could not be decoded as UTF-8.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve file: {exc}",
        ) from exc


@router.put("/files/{file_path:path}")
async def update_file_content(
    file_path: str,
    request: FileUpdateRequest,
    current_business: Business = Depends(get_current_business),
) -> Dict[str, str]:
    ext = Path(file_path).suffix.lower()
    if ext not in [".txt", ".md"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt and .md files can be modified.",
        )

    content_bytes = request.content.encode("utf-8")
    mime_type = "text/plain" if ext == ".txt" else "text/markdown"

    # TODO: Should we allow the creation of new files if a file doesn't exist?
    # Since the GET operation only checks this
    # temp workaround
    if file_path not in s3_storage.list_files(bucket=current_business.business_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_path}' not found. Only existing files can be updated.",
        )

    await _process_deletion(
        business_id=current_business.business_id,
        file_path=file_path,
    )

    await _process_upload(
        business_id=current_business.business_id,
        file_path=file_path,
        content=content_bytes,
        mime_type=mime_type,
    )

    return {"message": f"File '{file_path}' updated and re-indexed successfully."}


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
