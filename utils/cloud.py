"""
Cloud helpers (Cloud SQL, Secret Manager, GCS) shared with vizon_classifier_embedding's conventions.

- Cloud SQL: connection info (USER/DB_PASS/INSTANCE/DB) is fetched from Secret Manager on every
  run (`dev_pg_db` / `prd_pg_db`); nothing is stored on disk or in env vars.
- GCS: downloads preserve the blob's own filename, so the shared image folder is reusable by
  every repo that points at it.

Heavy google-cloud imports are lazy so importing this module is free for non-cloud workflows.
"""

import json
from pathlib import Path
from typing import List, Tuple

SECRET_PROJECT_ID = "655525621458"
_SECRET_NAMES = {"dev": "dev_pg_db", "prd": "prd_pg_db"}

_connector = None
_connections = {}
_storage_client = None


def get_secret(secret_name: str) -> str:
    from google.cloud import secretmanager

    # REST transport avoids a grpc c-ares DNS flake seen on some networks (same as the
    # embedding repo).
    client = secretmanager.SecretManagerServiceClient(transport="rest")
    name = f"projects/{SECRET_PROJECT_ID}/secrets/{secret_name}/versions/latest"
    return client.access_secret_version(request={"name": name}).payload.data.decode("UTF-8")


def get_connection(env: str):
    """Return a cached Cloud SQL (Postgres, pg8000) connection for env."""
    global _connector
    if env not in _SECRET_NAMES:
        raise ValueError(f"Unsupported env {env!r} - expected one of {sorted(_SECRET_NAMES)}")

    connection = _connections.get(env)
    if connection is None or connection._usock is None:
        from google.cloud.sql.connector import Connector

        if _connector is None:
            _connector = Connector()
        creds = json.loads(get_secret(_SECRET_NAMES[env]))
        connection = _connector.connect(
            creds["INSTANCE"], "pg8000", user=creds["USER"], password=creds["DB_PASS"], db=creds["DB"]
        )
        _connections[env] = connection
    return connection


def execute_query(query: str, params: tuple = (), env: str = "dev"):
    """
    Run a query. Returns the rows as a JSON string (list of column -> value objects) for a
    SELECT, otherwise commits and returns True.
    """
    connection = get_connection(env)
    cursor = connection.cursor()
    try:
        cursor.execute(query, params)
        if cursor.description is None:
            connection.commit()
            return True
        columns = [c[0] for c in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        cursor.close()
    connection.commit()
    return json.dumps(rows, default=str)


def _client():
    global _storage_client
    if _storage_client is None:
        from google.cloud import storage

        _storage_client = storage.Client()
    return _storage_client


def parse_gcs_path(gcs_path: str) -> Tuple[str, str]:
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Not a valid GCS path (must start with 'gs://'): {gcs_path}")
    bucket, _, blob = gcs_path[len("gs://"):].partition("/")
    if not bucket:
        raise ValueError(f"Missing bucket name in GCS path: {gcs_path}")
    return bucket, blob


def list_blobs(gcs_path: str) -> List[str]:
    """Full gs:// paths of every object under the prefix (folder placeholders excluded)."""
    bucket_name, prefix = parse_gcs_path(gcs_path)
    return [
        f"gs://{bucket_name}/{b.name}"
        for b in _client().bucket(bucket_name).list_blobs(prefix=prefix)
        if not b.name.endswith("/")
    ]


def download_blob(gcs_path: str, destination) -> Path:
    bucket_name, blob_path = parse_gcs_path(gcs_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".part")
    _client().bucket(bucket_name).blob(blob_path).download_to_filename(str(tmp))
    tmp.replace(destination)  # atomic: never leave a truncated file in the shared folder
    return destination
