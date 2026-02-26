from functools import lru_cache
from app.service import HybridRetrievalService

@lru_cache()
def get_service() -> HybridRetrievalService:
    return HybridRetrievalService()
