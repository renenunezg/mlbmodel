import os
from sqlalchemy import create_engine, event
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in .env")

engine = create_engine(DATABASE_URL)

_WRITE_KEYWORDS = ("insert", "update", "delete", "truncate", "create", "drop", "alter", "replace", "merge")


def _is_write_statement(statement: str) -> bool:
    return statement.lstrip().lower().startswith(_WRITE_KEYWORDS)


def writes_allowed() -> bool:
    """DATABASE_URL is the live production Supabase, not a dev copy, so a bare
    local run is denied by default. CI opts in automatically (GitHub Actions
    sets GITHUB_ACTIONS=true); a human sets MLBMODEL_DB_WRITES=1 for a backfill."""
    return os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("MLBMODEL_DB_WRITES") == "1"


@event.listens_for(engine, "before_cursor_execute")
def _block_unauthorized_writes(conn, cursor, statement, parameters, context, executemany):
    if _is_write_statement(statement) and not writes_allowed():
        raise RuntimeError(
            "Refusing to write to the production database (DATABASE_URL is the live Supabase). "
            "Re-run with MLBMODEL_DB_WRITES=1 to mutate production intentionally. "
            "CI runs are allowed automatically."
        )
