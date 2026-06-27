from dotenv import load_dotenv
import os
import psycopg

load_dotenv()

def get_connection_params():
    return {
        "host": os.getenv("SUPABASE_DB_HOST"),
        "port": os.getenv("SUPABASE_DB_PORT", "5432"),
        "dbname": os.getenv("SUPABASE_DB_NAME", "postgres"),
        "user": os.getenv("SUPABASE_DB_USER"),
        "password": os.getenv("SUPABASE_DB_PASSWORD"),
        "sslmode": "require",
    }

def test_connection():
    params = get_connection_params()
    missing = [k for k, v in params.items() if k != "sslmode" and not v]
    if missing:
        return {"status": "error", "detail": "Faltan variables: " + ", ".join(missing)}

    try:
        with psycopg.connect(**params) as conn:
            with conn.cursor() as cur:
                cur.execute("select version();")
                version = cur.fetchone()[0]
        return {"status": "ok", "detail": "Conexion exitosa", "db_version": version}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    

def get_clientes(limit: int = 20):
    params = get_connection_params()
    missing = [k for k, v in params.items() if k != "sslmode" and not v]
    if missing:
        raise ValueError("Faltan variables: " + ", ".join(missing))
    
    query = '''
    select
        customerid,
        gender,
        seniorcitizen,
        partner,
        dependents,
        tenure,
        phoneservice,
        multiplelines,
        internetservice,
        onlinesecurity,
        onlinebackup,
        deviceprotection,
        techsupport,
        streamingtv,
        streamingmovies,
        contract,
        paperlessbilling,
        paymentmethod,
        monthlycharges,
        totalcharges,
        churn
    from public.telco_mvp   
    order by customerid
    limit %s;
    '''

    with psycopg.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]

    results = []
    for row in rows:
        item = {}
        for col, value in zip(columns, row):
            if hasattr(value, "isoformat"):
                item[col] = value.isoformat()
            else:
                item[col] = value
        results.append(item)

    return results