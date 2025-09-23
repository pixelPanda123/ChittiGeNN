from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Lazy import to avoid circular dependency issues
from app.services.search_engine import search_engine
from app.database import get_db

# ------------------------------
# Pydantic model for search request
# ------------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10


# ------------------------------
# APIRouter
# ------------------------------
router = APIRouter(
    prefix="/search",
    tags=["search"]
)

# ------------------------------
# POST endpoint
# ------------------------------
@router.post("/")
async def search_post(req: SearchRequest, db: Session = Depends(get_db)):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    results = search_engine.search(db, query, req.top_k)
    return {"results": results}


# ------------------------------
# GET endpoint with `query` param
# ------------------------------
@router.get("/")
async def search_get(query: str, top_k: Optional[int] = 10, db: Session = Depends(get_db)):
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query is required")
    
    results = search_engine.search(db, q, top_k)
    return {"results": results}


# ------------------------------
# GET endpoint with `q` param (alternate)
# ------------------------------
@router.get("/query")
async def search_query(q: str, top_k: Optional[int] = 10, db: Session = Depends(get_db)):
    qq = (q or "").strip()
    if not qq:
        raise HTTPException(status_code=400, detail="`q` parameter is required")
    
    results = search_engine.search(db, qq, top_k)
    return {"results": results}
