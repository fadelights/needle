from typing import List, Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    message: str
    business_id: str


class SourceChunk(BaseModel):
    content: str
    score: Optional[float] = None
    file_path: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    business_id: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[SourceChunk]
