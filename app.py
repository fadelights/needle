from contextlib import asynccontextmanager

from fastapi import FastAPI

from balequeue.api import router
from balequeue.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    pass


app = FastAPI(
    title="Balequeue QA Service",
    description="Intelligent business QA powered by Haystack.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router, prefix="/api")


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "balequeue"}


if __name__ == "__main__":
    import uvicorn

    # TODO: 0.0.0.0 is required for running in Docker
    uvicorn.run(app, host="127.0.0.1", port=settings.balequeue_port)
