import threading
from typing import Optional
from app.service import HybridRetrievalService

_service: Optional[HybridRetrievalService] = None
_lock = threading.Lock()

def get_service() -> HybridRetrievalService:
    # FastAPI runs sync dependencies in a threadpool; lru_cache does not stop
    # concurrent first calls from each building the model in parallel.
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                _service = HybridRetrievalService()
    return _service
