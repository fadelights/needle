from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from needle.api import router
from needle.config import settings
from needle.database import Base, engine
from needle.models import (
    Business,
)  # noqa: F401 - This import is used for Base.metadata.create_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    pass


app = FastAPI(
    title="Needle QA Service",
    description="Intelligent business QA powered by Haystack.",
    version="0.2.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "needle"}


if __name__ == "__main__":
    import uvicorn

    # TODO: Apply security measures
    uvicorn.run(app, host="0.0.0.0", port=settings.needle_port)
