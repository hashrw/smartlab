import os
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine

ALLOWED_TABLES = [
    "pacientes",
    "diagnosticos",
    "sintomas",
    "paciente_sintoma",
    "organo_paciente",
]


def build_sql_engine():
    engine = create_engine(os.getenv("DB_URL_READONLY"))

    sql_db = SQLDatabase(engine)

    return NLSQLTableQueryEngine(
        sql_database=sql_db,
        tables=ALLOWED_TABLES,
    )