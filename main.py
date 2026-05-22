"""
GenVR Workflow Engine — entry point.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="GenVR Workflow Engine",
    description=(
        "A Python-powered workflow engine that exposes pluggable nodes over HTTP. "
        "Designed to integrate with the GenVR Workflow Designer."
    ),
    version="0.1.0",
)

# Allow the GenVR web app (any origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "service": "GenVR Workflow Engine"}
