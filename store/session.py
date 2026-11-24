from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from config import settings
from store.model import Base

_ENGINE_CACHE: dict[str, Engine] = {}
_SESSION_FACTORY_CACHE: dict[str, sessionmaker] = {}


def configure_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=FULL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
    finally:
        cursor.close()


def get_engine(db_path: str | Path | None = None) -> Engine:
    resolved = str(
        Path(db_path).resolve()
        if db_path is not None
        else settings.storage.sqlite_db_path.resolve()
    )
    if resolved not in _ENGINE_CACHE:
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{resolved}",
            connect_args={"check_same_thread": False, "timeout": 30},
            future=True,
        )
        event.listen(engine, "connect", configure_sqlite_pragmas)
        _ENGINE_CACHE[resolved] = engine
    return _ENGINE_CACHE[resolved]


def get_session_factory(db_path: str | Path | None = None) -> sessionmaker:
    resolved = str(
        Path(db_path).resolve()
        if db_path is not None
        else settings.storage.sqlite_db_path.resolve()
    )
    if resolved not in _SESSION_FACTORY_CACHE:
        engine = get_engine(resolved)
        _SESSION_FACTORY_CACHE[resolved] = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False
        )
    return _SESSION_FACTORY_CACHE[resolved]


def init_store(db_path: str | Path | None = None) -> None:
    engine = get_engine(db_path)
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    collection_name UNINDEXED,
                    source,
                    symbol_name,
                    content,
                    tokenize = 'porter unicode61 tokenchars ''_'''
                );
                """
            )
        )


@contextmanager
def get_session(db_path: str | Path | None = None) -> Generator[Session, None, None]:
    init_store(db_path)
    factory = get_session_factory(db_path)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
