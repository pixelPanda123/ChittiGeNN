from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Path to your existing database
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/chittigen.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
