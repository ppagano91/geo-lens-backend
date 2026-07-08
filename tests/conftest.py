from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.session import get_db
from app.main import app

VALID_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-58.40, -34.60],
            [-58.38, -34.60],
            [-58.38, -34.62],
            [-58.40, -34.62],
            [-58.40, -34.60],
        ]
    ],
}


def _database_available() -> bool:
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


DATABASE_AVAILABLE = _database_available()

requires_database = pytest.mark.skipif(
    not DATABASE_AVAILABLE,
    reason="PostgreSQL/PostGIS is not available. Start PostgreSQL local and run migrations.",
)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection, autocommit=False, autoflush=False)()
    session.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(session: Session, trans) -> None:
        if trans.nested and not trans._parent.nested:
            session.begin_nested()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        engine.dispose()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def override_get_db() -> Generator[Session, None, None]:
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
