"""Execute and print all analytical queries in sql/queries.sql."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "processed" / "predictive_maintenance.sqlite3"
DEFAULT_QUERIES = ROOT / "sql" / "queries.sql"


def read_sql_statements(path: Path) -> list[str]:
    """Split a SQL file into complete SQLite statements."""
    statements: list[str] = []
    buffer = ""
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement:
                statements.append(statement)
            buffer = ""
    if buffer.strip():
        raise ValueError(f"Incomplete SQL statement at end of {path}")
    return statements


def run_queries(db_path: Path, queries_path: Path, row_limit: int = 8) -> int:
    """Execute all queries, print a small preview, and return statement count."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}. Run scripts/build_database.py first.")
    statements = read_sql_statements(queries_path)
    with sqlite3.connect(db_path) as connection:
        for index, statement in enumerate(statements, start=1):
            cursor = connection.execute(statement)
            rows = cursor.fetchmany(row_limit)
            print(f"\n--- Query {index} ---")
            if cursor.description:
                print(" | ".join(column[0] for column in cursor.description))
                for row in rows:
                    print(" | ".join(str(value) for value in row))
    return len(statements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--row-limit", type=int, default=8)
    args = parser.parse_args()
    count = run_queries(args.db, args.queries, args.row_limit)
    print(f"\nExecuted {count} analytical SQL queries successfully.")
