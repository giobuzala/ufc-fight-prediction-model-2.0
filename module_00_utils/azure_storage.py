from __future__ import annotations

import os
from io import BytesIO, StringIO
from typing import Optional

import pandas as pd
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Connection string is always read from Key Vault; env is not used.
KEY_VAULT_NAME = "kv-azure-lab-ufc"
KEY_VAULT_SECRET_NAME = "AZURE-STORAGE-CONNECTION-STRING"

_container_client = None


def _get_connection_string() -> str:
    """
    Get Azure Storage connection string from Key Vault only.

    Uses KEY_VAULT_NAME and KEY_VAULT_SECRET_NAME with DefaultAzureCredential.
    Environment variables are not used for the connection string.
    """
    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        kv_uri = f"https://{KEY_VAULT_NAME}.vault.azure.net"
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=kv_uri, credential=credential)
        secret = client.get_secret(KEY_VAULT_SECRET_NAME)
        value = (secret.value or "").strip()
        if not value:
            raise ValueError("Key Vault secret is empty")
        return value
    except Exception as e:
        raise ValueError(
            f"Could not read connection string from Key Vault '{KEY_VAULT_NAME}' "
            f"(secret '{KEY_VAULT_SECRET_NAME}'): {e}. "
            "Ensure the vault exists and the app has access."
        ) from e


def _get_container_client():
    """
    Lazily initialize Azure Blob container client.

    Connection string is always from Key Vault. Container name may come from
    AZURE_STORAGE_CONTAINER env (default 'pipeline-data').
    """
    global _container_client
    if _container_client is not None:
        return _container_client

    load_dotenv()
    connection_string = _get_connection_string()

    container_name = os.getenv("AZURE_STORAGE_CONTAINER", "pipeline-data")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    _container_client = blob_service_client.get_container_client(container_name)
    try:
        _container_client.create_container()
    except ResourceExistsError:
        pass
    return _container_client


# Helpers


def list_blobs(prefix: Optional[str] = None) -> list[str]:
    container_client = _get_container_client()
    blobs = container_client.list_blobs(name_starts_with=prefix)
    return [blob.name for blob in blobs]


def read_csv_from_azure(blob_path: str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file from Azure Blob Storage into a pandas DataFrame.
    """
    container_client = _get_container_client()
    blob_client = container_client.get_blob_client(blob_path)

    if not blob_client.exists():
        available = list_blobs("/".join(blob_path.split("/")[:-1]))
        raise FileNotFoundError(
            f"Blob not found: {blob_path}\n"
            f"Available blobs under parent path:\n" + "\n".join(available)
        )

    blob_data = blob_client.download_blob().readall()
    return pd.read_csv(BytesIO(blob_data), **kwargs)


def write_csv_to_azure(df: pd.DataFrame, blob_path: str, index: bool = False, **kwargs) -> None:
    """
    Write a pandas DataFrame to Azure Blob Storage as CSV.

    Parameters
    ----------
    df
        DataFrame to write.
    blob_path
        Path within the container, e.g.
        'module_01_scrapers/output/cleaned_fights.csv'
    index
        Whether to include the DataFrame index.
    **kwargs
        Additional arguments passed to DataFrame.to_csv().
    """
    buffer = StringIO()
    df.to_csv(buffer, index=index, **kwargs)

    container_client = _get_container_client()
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(buffer.getvalue(), overwrite=True)


def read_parquet_from_azure(blob_path: str, **kwargs) -> pd.DataFrame:
    """
    Read a Parquet file from Azure Blob Storage into a pandas DataFrame.
    """
    container_client = _get_container_client()
    blob_client = container_client.get_blob_client(blob_path)

    if not blob_client.exists():
        available = list_blobs("/".join(blob_path.split("/")[:-1]))
        raise FileNotFoundError(
            f"Blob not found: {blob_path}\n"
            f"Available blobs under parent path:\n" + "\n".join(available)
        )

    blob_data = blob_client.download_blob().readall()
    return pd.read_parquet(BytesIO(blob_data), **kwargs)


def write_parquet_to_azure(df: pd.DataFrame, blob_path: str, index: bool = False, **kwargs) -> None:
    """
    Write a pandas DataFrame to Azure Blob Storage as Parquet.
    """
    buffer = BytesIO()
    df.to_parquet(buffer, index=index, **kwargs)
    buffer.seek(0)

    container_client = _get_container_client()
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(buffer.getvalue(), overwrite=True)


def upload_file_to_azure(local_path: str, blob_path: str) -> None:
    """
    Upload a local file to Azure Blob Storage.
    """
    container_client = _get_container_client()
    blob_client = container_client.get_blob_client(blob_path)

    with open(local_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)


def download_file_from_azure(blob_path: str, local_path: str) -> None:
    """
    Download a blob from Azure Blob Storage to a local file.
    """
    container_client = _get_container_client()
    blob_client = container_client.get_blob_client(blob_path)
    data = blob_client.download_blob().readall()

    with open(local_path, "wb") as f:
        f.write(data)
