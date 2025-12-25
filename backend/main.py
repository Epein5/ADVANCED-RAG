from fastapi import FastAPI
from backend.core.config import config
from backend.core.logging import setup_logging
from backend.api.v1.injgestion import router as ingestion_router


setup_logging()

app = FastAPI(
    title=config.app_name,
    version="1.0.0",
    docs_url='/dicks',
)

app.include_router(ingestion_router, prefix="/api/v1/ingestion", tags=["ingestion"])
