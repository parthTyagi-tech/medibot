import logging
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)


def run_migrations(engine=None):
    """
    Auto-migrate schema changes that SQLAlchemy's db.create_all() does not apply
    to pre-existing database tables (such as PostgreSQL on Render).
    
    This function is idempotent and safe to run on every application startup.
    """
    if engine is None:
        from research.src.auth import db
        engine = db.engine

    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    # 1. chat_session table migrations
    if "chat_session" in existing_tables:
        existing_cols = {col["name"] for col in inspector.get_columns("chat_session")}
        if "patient_state_json" not in existing_cols:
            logger.info("Migrating: Adding 'patient_state_json' column to 'chat_session' table...")
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE chat_session ADD COLUMN patient_state_json TEXT DEFAULT ''"))
            logger.info("Migration: 'patient_state_json' added successfully.")
        else:
            logger.info("Column 'patient_state_json' already exists on 'chat_session'.")

    # 2. user table migrations (ensure memory, reset_otp, otp_expiry exist)
    if "user" in existing_tables:
        existing_cols = {col["name"] for col in inspector.get_columns("user")}
        with engine.begin() as conn:
            if "memory" not in existing_cols:
                logger.info("Migrating: Adding 'memory' column to 'user' table...")
                conn.execute(text('ALTER TABLE "user" ADD COLUMN memory TEXT DEFAULT \'\''))
            if "reset_otp" not in existing_cols:
                logger.info("Migrating: Adding 'reset_otp' column to 'user' table...")
                conn.execute(text('ALTER TABLE "user" ADD COLUMN reset_otp VARCHAR(10)'))
            if "otp_expiry" not in existing_cols:
                logger.info("Migrating: Adding 'otp_expiry' column to 'user' table...")
                conn.execute(text('ALTER TABLE "user" ADD COLUMN otp_expiry TIMESTAMP'))


if __name__ == "__main__":
    import os
    import sys
    # Add project root to sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app import app
    with app.app_context():
        run_migrations()
        print("Database migration completed successfully.")
