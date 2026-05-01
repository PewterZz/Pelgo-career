from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(candidates_router, prefix="/api/v1")
app.include_router(matches_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")

_static = Path(__file__).resolve().parent.parent.parent / "static"
if _static.is_dir():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
