"""Storage utilities for local filesystem and future S3 support."""
import uuid
from pathlib import Path
from typing import Union
import shutil

from .config import settings


def new_id() -> str:
    """Generate a new unique ID."""
    return str(uuid.uuid4())


class LocalStorage:
    """Local filesystem storage backend."""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or settings.DATA_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, file_path: Union[str, Path], key: str) -> str:
        """Save a file to storage and return the storage key."""
        dest = self.base_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        return key

    def save_bytes(self, data: bytes, key: str, content_type: str = None) -> str:
        """Save bytes to storage and return the storage key."""
        dest = self.base_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return key

    def get_path(self, key: str) -> Path:
        """Get the local filesystem path for a storage key."""
        return self.base_dir / key

    def get_url(self, key: str) -> str:
        """Get a URL for a storage key (for local, returns file path)."""
        return f"file://{self.get_path(key)}"

    def exists(self, key: str) -> bool:
        """Check if a file exists in storage."""
        return self.get_path(key).exists()

    def delete(self, key: str) -> bool:
        """Delete a file from storage."""
        path = self.get_path(key)
        if path.exists():
            path.unlink()
            return True
        return False


# Global storage instance
storage = LocalStorage()
