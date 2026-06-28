from dotenv import load_dotenv
from datetime import datetime
import os
import pandas as pd
import psycopg
from psycopg import sql

load_dotenv()

CSV_PATH = "data/processed/telco_limpio.csv"
SHEET_NAME = "Sheet1"   
TABLE_NAME = "public.telco_mvp"

# Modo destructivo:
# True  -> elimina la tabla destino, la vuelve a crear según la estructura del Excel e inserta los datos.
# False -> conserva la tabla actual y solo controla si se limpian o agregan registros.
DROP_AND_RECREATE_TABLE = True

# Solo se usa cuando DROP_AND_RECREATE_TABLE = False.
# True  -> borra los registros actuales con TRUNCATE antes de insertar.
# False -> agrega los datos al final de la tabla existente.
CLEAR_BEFORE_INSERT = False

def get_connection_params():
    """
    Obtiene los parámetros de conexión a PostgreSQL desde variables de entorno.
    """
    return {
        "host": os.getenv("SUPABASE_DB_HOST"),
        "port": os.getenv("SUPABASE_DB_PORT", "5432"),
        "dbname": os.getenv("SUPABASE_DB_NAME", "postgres"),
        "user": os.getenv("SUPABASE_DB_USER"),
        "password": os.getenv("SUPABASE_DB_PASSWORD"),
        "sslmode": "require",
    }

def validate_env():
    """
    Valida que existan las variables de entorno requeridas para la conexión.
    """
    params = get_connection_params()
    missing = [key for key, value in params.items() if key != "sslmode" and not value]

    if missing:
        raise ValueError("Faltan variables de entorno: " + ", ".join(missing))

    return params

params = validate_env()

safe_params = {
    key: ("***" if key == "password" else value)
    for key, value in params.items()
}

safe_params

def load_dataframe():
    """
    Lee directamente el archivo CSV limpio.

    Retorna
    -------
    pandas.DataFrame
        Datos leídos desde CSV, sin transformaciones adicionales.
    """
    return pd.read_excel(CSV_PATH)

def table_identifier(table_name):
    """
    Convierte un nombre de tabla tipo 'schema.tabla' en un identificador SQL seguro.
    """
    return sql.Identifier(*table_name.split("."))


def column_identifiers(columns):
    """
    Convierte la lista de columnas del DataFrame en identificadores SQL seguros.
    """
    return sql.SQL(", ").join(sql.Identifier(str(column)) for column in columns)

def infer_postgres_type(series):
    """
    Infere un tipo PostgreSQL básico a partir del tipo de dato detectado por pandas.
    No modifica los valores de la columna.
    """
    dtype = series.dtype

    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"

    return "TEXT"


def build_schema_report(df):
    """
    Construye un reporte informativo con la estructura que se usará para crear la tabla.
    """
    return pd.DataFrame({
        "columna_excel": [str(column) for column in df.columns],
        "tipo_pandas_detectado": [str(df[column].dtype) for column in df.columns],
        "tipo_postgresql_sugerido": [infer_postgres_type(df[column]) for column in df.columns],
    })

def clear_table(conn):
    """
    Elimina todos los registros de la tabla destino y reinicia el identificador autoincremental.
    No elimina la estructura de la tabla.
    """
    query = sql.SQL("truncate table {} restart identity;").format(
        table_identifier(TABLE_NAME)
    )

    with conn.cursor() as cur:
        cur.execute(query)


def drop_table(conn):
    """
    Elimina la tabla destino si existe.
    Advertencia: esta acción elimina la estructura y los registros de la tabla.
    """
    query = sql.SQL("drop table if exists {};").format(
        table_identifier(TABLE_NAME)
    )

    with conn.cursor() as cur:
        cur.execute(query)


def create_table_from_dataframe(conn, df):
    """
    Crea la tabla destino usando las columnas y tipos detectados desde el DataFrame.
    No modifica los datos del DataFrame.
    """
    if df.empty:
        raise ValueError("No se puede crear la tabla porque el DataFrame está vacío.")

    column_definitions = []
    for column in df.columns:
        pg_type = infer_postgres_type(df[column])
        column_definitions.append(
            sql.SQL("{} {}").format(
                sql.Identifier(str(column)),
                sql.SQL(pg_type)
            )
        )

    query = sql.SQL("create table {} ({});").format(
        table_identifier(TABLE_NAME),
        sql.SQL(", ").join(column_definitions)
    )

    with conn.cursor() as cur:
        cur.execute(query)

def insert_dataframe(conn, df):
    """
    Inserta un DataFrame en la tabla destino sin modificar sus datos.
    """
    columns = list(df.columns)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in columns)

    query = sql.SQL("insert into {} ({}) values ({})").format(
        table_identifier(TABLE_NAME),
        column_identifiers(columns),
        placeholders,
    )

    rows = list(df.itertuples(index=False, name=None))

    with conn.cursor() as cur:
        cur.executemany(query, rows)

    return len(rows)

def add_report_event(report, process_start, step, status="OK", detail="", step_start=None):
    """
    Agrega un evento al reporte de ejecución y lo imprime en pantalla.
    """
    now = datetime.now()
    elapsed_seconds = round((now - process_start).total_seconds(), 2)
    duration_seconds = None

    if step_start is not None:
        duration_seconds = round((now - step_start).total_seconds(), 2)

    report.append({
        "momento": now.strftime("%Y-%m-%d %H:%M:%S"),
        "segundos_desde_inicio": elapsed_seconds,
        "duracion_paso_segundos": duration_seconds,
        "paso": step,
        "estado": status,
        "detalle": detail,
    })

    print(f"[{status}] {step}: {detail}")

def run_load_process():
    """
    Ejecuta el proceso completo de carga y retorna el número de filas insertadas
    junto con un DataFrame de reporte de ejecución.
    """
    report = []
    process_start = datetime.now()
    inserted_rows = 0

    try:
        add_report_event(
            report,
            process_start,
            step="Inicio del proceso",
            detail="Se inicia la carga directa hacia Supabase/PostgreSQL."
        )

        step_start = datetime.now()
        params = validate_env()
        add_report_event(
            report,
            process_start,
            step="Validación de variables de entorno",
            detail="Credenciales disponibles en archivo .env.",
            step_start=step_start
        )

        step_start = datetime.now()
        df = load_dataframe()
        add_report_event(
            report,
            process_start,
            step="Lectura del archivo Excel",
            detail=f"Archivo leído correctamente: {len(df)} filas y {len(df.columns)} columnas.",
            step_start=step_start
        )

        step_start = datetime.now()
        schema_report = build_schema_report(df)
        add_report_event(
            report,
            process_start,
            step="Lectura de estructura del Excel",
            detail=f"Estructura detectada para {len(schema_report)} columnas.",
            step_start=step_start
        )

        step_start = datetime.now()
        with psycopg.connect(**params) as conn:
            add_report_event(
                report,
                process_start,
                step="Conexión a Supabase/PostgreSQL",
                detail="Conexión establecida correctamente.",
                step_start=step_start
            )

            if DROP_AND_RECREATE_TABLE:
                step_start = datetime.now()
                drop_table(conn)
                create_table_from_dataframe(conn, df)
                add_report_event(
                    report,
                    process_start,
                    step="Recreación de tabla destino",
                    detail=f"Tabla {TABLE_NAME} eliminada y creada nuevamente según estructura del Excel.",
                    step_start=step_start
                )
            elif CLEAR_BEFORE_INSERT:
                step_start = datetime.now()
                clear_table(conn)
                add_report_event(
                    report,
                    process_start,
                    step="Limpieza de tabla destino",
                    detail=f"Tabla {TABLE_NAME} limpiada con TRUNCATE antes de insertar.",
                    step_start=step_start
                )
            else:
                add_report_event(
                    report,
                    process_start,
                    step="Modo de carga incremental",
                    detail="No se limpia ni recrea la tabla; los registros se agregarán al final."
                )

            step_start = datetime.now()
            inserted_rows = insert_dataframe(conn, df)
            add_report_event(
                report,
                process_start,
                step="Inserción de registros",
                detail=f"Se insertaron {inserted_rows} filas en {TABLE_NAME}.",
                step_start=step_start
            )

            step_start = datetime.now()
            conn.commit()
            add_report_event(
                report,
                process_start,
                step="Confirmación de transacción",
                detail="Commit ejecutado correctamente.",
                step_start=step_start
            )

        add_report_event(
            report,
            process_start,
            step="Fin del proceso",
            detail="Carga completada exitosamente."
        )

    except Exception as error:
        add_report_event(
            report,
            process_start,
            step="Error del proceso",
            status="ERROR",
            detail=str(error)
        )
        print(pd.DataFrame(report))
        raise

    return inserted_rows, pd.DataFrame(report)


inserted_rows, execution_report_df = run_load_process()
print(execution_report_df)

