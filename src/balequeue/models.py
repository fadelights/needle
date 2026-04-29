import uuid
from sqlalchemy import Column, Integer, String
from .database import Base


class Business(Base):
    __tablename__ = "businesses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    business_id = Column(
        String, unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
