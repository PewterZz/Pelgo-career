from contextlib import asynccontextmanager

from fastapi import FastAPI

from pelgo.api.routes.admin import router as admin_router
from pelgo.api.routes.candidates import router as candidates_router
from pelgo.api.routes.matches import router as matches_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Pelgo Career Intelligence API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(candidates_router, prefix="/api/v1")
app.include_router(matches_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
