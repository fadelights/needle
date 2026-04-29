from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .config import settings
from .database import SessionLocal
from .models import Business

passwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


def get_passwd_hash(password):
    return passwd_context.hash(password)


def verify_passwd(plain, hash):
    return passwd_context.verify(plain, hash)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    if expires_delta:
        expiration = datetime.now() + expires_delta
    else:
        expiration = datetime.now() + timedelta(minutes=15)

    to_encode = data.copy()
    to_encode.update({"exp": expiration})
    encoded_jwt = jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.auth_algorithm
    )
    return encoded_jwt


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_business(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> Business:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.auth_algorithm]
        )
        business_name: str = payload.get("sub")
        if business_name is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    business = db.query(Business).filter(Business.name == business_name).first()
    if business is None:
        raise credentials_exception
    return business
