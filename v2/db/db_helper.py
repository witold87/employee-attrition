import psycopg2
from v2.config.app_config import get_db_config
from sqlalchemy import create_engine


def create_conn():
    params = get_db_config()
    return psycopg2.connect(**params)


def with_connection(func):
    def with_connection_(*args, **kwargs):
        params = get_db_config()
        conn = psycopg2.connect(**params)
        try:
            rv = func(conn, *args, **kwargs)
        except Exception as e:
            conn.rollback()
            print(f'SQL failed fue to {e}')
        else:
            conn.commit()
        finally:
            conn.close()
        return rv
    return with_connection_


def connect_database(func):
    def wrapper(*args, **kwargs):
        # Connect to the database
        params = get_db_config()
        conn = psycopg2.connect(**params)
        cursor = conn.cursor()

        # Call the function with the database connection
        result = func(cursor, *args, **kwargs)

        # Commit changes and close the connection
        conn.commit()
        conn.close()

        return result

    return wrapper

def query_data(query):
    engine = connect_to_db()
    conn = engine.connect()
    results = conn.execute(query)
    return results

def get_max_inference_date():
    results = query_data('select max(inference_date) from public.predictions_with_explainers')
    max_date = None
    for row in results:
        max_date = row[0]
    return max_date

def connect_to_db():
    params = get_db_config()
    engine = create_engine(f'postgresql+psycopg2://{params["user"]}:{params["password"]}@{params["host"]}/{params["database"]}')
    return engine