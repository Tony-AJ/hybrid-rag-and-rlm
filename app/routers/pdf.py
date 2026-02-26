from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from app.models import HybridSearchResult
from app.service import HybridRetrievalService
from app.dependencies import get_service

router = APIRouter(prefix="/pdf", tags=["pdf"])

@router.post("/search", response_model=HybridSearchResult)
def pdf_search(
    file: UploadFile = File(...),
    query: str = Form(...),
    service: HybridRetrievalService = Depends(get_service)
):
    result = service.process_pdf_and_search(file, query)
    if not result:
        raise HTTPException(status_code=404, detail="No results found in the PDF")
    return result
