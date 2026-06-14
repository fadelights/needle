from datetime import timedelta
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from .auth import create_access_token, get_current_business, get_db, get_passwd_hash, verify_passwd
from .config import settings
from .core import delete, get_content, index, list_, query, update_content
from .models import Business
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

# TODO: Are the pipelines thread-safe?


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/auth/register", response_model=BusinessOut, status_code=status.HTTP_201_CREATED)
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
    business_id: str,
    original_path: str,
    content: bytes,
    mime_type: str,
    storage_path: str = None,
) -> None:
    """
    Helper function to upload file to S3 and run the indexing pipeline.
    Used by both upload and update endpoints.
    """
    try:
        index(
            business_id=business_id,
            content=content,
            mime_type=mime_type,
            original_path=original_path,
            storage_path=storage_path,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload and index document: {exc}",
        ) from exc


async def _process_deletion(business_id: str, storage_path: str) -> Dict[str, str]:
    """
    Helper function to delete file from S3 and document store.
    Used by both delete and update endpoints.
    """
    try:
        return delete(business_id=business_id, storage_path=storage_path)
    except FileNotFoundError:
        # TODO: If the file is already missing from storage, we should still attempt to delete any documents that reference it.
        return {"message": f"Document '{storage_path}' does not exist."}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {exc}",
        ) from exc


@router.get("/files", response_model=List[Dict])
async def list_files(
    current_business: Business = Depends(get_current_business),
) -> List[Dict]:
    try:
        return list_(business_id=current_business.business_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {exc}",
        ) from exc


# TODO: Handle duplicate uploads
@router.post(
    "/files",
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
        original_path=file.filename,
        content=file_content,
        mime_type=file.content_type,
    )

    return UploadResponse(
        message="Document uploaded successfully.",
        business_id=current_business.business_id,
    )


@router.get("/files/{storage_path:path}", response_model=FileContentResponse)
async def get_file_content(
    storage_path: str,
    current_business: Business = Depends(get_current_business),
) -> FileContentResponse:
    try:
        content = get_content(business_id=current_business.business_id, storage_path=storage_path)
        return FileContentResponse(content=content, storage_path=storage_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{storage_path}' not found.",
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{storage_path}' could not be decoded as UTF-8.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve file: {exc}",
        ) from exc


@router.delete("/files/{storage_path:path}")
async def delete_file(
    storage_path: str,
    current_business: Business = Depends(get_current_business),
) -> Dict[str, str]:
    return await _process_deletion(
        business_id=current_business.business_id,
        storage_path=storage_path,
    )


@router.put("/files/{storage_path:path}")
async def update_file_content(
    storage_path: str,
    request: FileUpdateRequest,
    current_business: Business = Depends(get_current_business),
) -> Dict[str, str]:
    try:
        return update_content(
            business_id=current_business.business_id,
            storage_path=storage_path,
            content=request.content,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{storage_path}' not found. Only existing files can be updated.",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update file: {exc}",
        ) from exc


@router.post("/query", response_model=QueryResponse)
async def query_business(
    request: QueryRequest,
    current_business: Business = Depends(get_current_business),
) -> QueryResponse:
    try:
        result = query(
            business_id=current_business.business_id,
            text=request.query,
            top_k=request.top_k,
            generate_response=request.generate_response,
        )

        source_chunks = [
            SourceChunk(
                content=chunk["content"],
                score=chunk["score"],
                storage_path=chunk["storage_path"],
            )
            for chunk in result["source_chunks"]
        ]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {exc}",
        ) from exc

    return QueryResponse(
        answer=result["answer"],
        source_chunks=source_chunks,
    )
