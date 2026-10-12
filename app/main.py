import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# Fix: ChromaDB 0.5.x calls posthog.capture(id, event, props) with 3 positional args
# but posthog v7+ only accepts capture(event, **kwargs) — signature mismatch
# We disable telemetry AND replace capture to prevent the TypeError
import posthog
posthog.disabled = True
posthog.capture = lambda *args, **kwargs: None

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import nodes, edges, search, pdf, graph, stats, rlm
from app.dependencies import get_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load embedding model + Chroma once at startup so startup fails loudly
    # instead of the first burst of requests racing to build it.
    get_service()
    yield

app = FastAPI(title="Hybrid Retrieval System", lifespan=lifespan)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Docker frontend
        "http://localhost:3001",    # Dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://frontend:80",      # Docker internal
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(nodes.router)
app.include_router(edges.router)
app.include_router(search.router)
app.include_router(pdf.router)
app.include_router(graph.router)
app.include_router(stats.router)
app.include_router(rlm.router)

@app.get("/")
def root():
    return {"message": "Hybrid Retrieval System API is running"}
