from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db

router = APIRouter(prefix="/health", tags=["health"]) 

@router.get("")
async def health(db: Session = Depends(get_db)):
    # simple db check
    try:
        db.execute("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok", "database": db_ok} 