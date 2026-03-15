from __future__ import annotations

import os
from io import BytesIO, StringIO
from typing import Optional

import pandas as pd
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


_container_client = None


def _get_container_client():
    """
    Lazily initialize Azure Blob container client.

    This keeps local-only runs working even when Azure env vars are not set,
    as long as Azure functions are not called.
    """
    global _container_client
    if _container_client is not None:
        return _container_client

    load_dotenv()
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER", "pipeline-data")

    if not connection_string:
        raise ValueError(
            "AZURE_STORAGE_CONNECTION_STRING is not set. "
            "Add it to your environment or .env file."
        )

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
