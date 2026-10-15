"""Database Configuration and SQLAlchemy Session Management."""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from backend.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite concurrency
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency yielding database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes tables in SQLite database and performs migrations for new columns."""
    from backend import models  # Ensure models are imported
    Base.metadata.create_all(bind=engine)

    # SQLite automatic schema migration for added columns
    with engine.connect() as conn:
        try:
            result = conn.exec_driver_sql("PRAGMA table_info(screenings)")
            existing_cols = {row[1] for row in result.fetchall()}

            new_columns = [
                ("lesion_count", "INTEGER DEFAULT 0"),
                ("overall_affected_area_percentage", "FLOAT"),
                ("valid_retinal_pixels", "INTEGER"),
                ("total_image_pixels", "INTEGER"),
                ("lesions_json", "TEXT DEFAULT '[]'"),
                ("class_summaries_json", "TEXT DEFAULT '{}'"),
                ("analyzed_image_path", "VARCHAR(255)"),
                ("primary_disease", "VARCHAR(128)"),
                ("disease_findings_json", "TEXT DEFAULT '[]'"),
            ]
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    conn.exec_driver_sql(f"ALTER TABLE screenings ADD COLUMN {col_name} {col_type}")
            conn.commit()
        except Exception:
            pass
