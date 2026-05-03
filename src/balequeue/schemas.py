from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class UploadResponse(BaseModel):
    message: str
    business_id: str


class SourceChunk(BaseModel):
    content: str
    score: Optional[float] = None
    file_path: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[SourceChunk]


class BusinessCreate(BaseModel):
    name: str
    password: str


class BusinessOut(BaseModel):
    name: str
    business_id: str

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str


class FileContentResponse(BaseModel):
    content: str
    file_path: str


class FileUpdateRequest(BaseModel):
    content: str
