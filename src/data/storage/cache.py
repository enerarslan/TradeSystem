"""
Data caching module for AlphaTrade system.

This module provides efficient data caching:
- Parquet-based storage for fast I/O
- Automatic cache invalidation
- Metadata tracking
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from config.settings import CACHE_DIR


class DataCache:
    """
    Efficient data cache using Parquet storage.

    Provides fast serialization/deserialization with
    automatic cache management and invalidation.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        max_age_hours: int = 24,
    ) -> None:
        """
        Initialize the cache.

        Args:
            cache_dir: Directory for cache files
            max_age_hours: Maximum cache age in hours
        """
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_hours = max_age_hours

        self._metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        """Load cache metadata from file."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        with open(self._metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def _generate_key(self, name: str, params: dict | None = None) -> str:
        """Generate a unique cache key."""
        key_data = name
        if params:
            key_data += json.dumps(params, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.parquet"

    def is_cached(
        self,
        name: str,
        params: dict | None = None,
    ) -> bool:
        """
        Check if data is cached and valid.

        Args:
            name: Cache entry name
            params: Optional parameters for unique key

        Returns:
            True if cached and not expired
        """
        key = self._generate_key(name, params)
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return False

        if key not in self._metadata:
            return False

        # Check age
        cached_time = datetime.fromisoformat(self._metadata[key]["created"])
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600

        return age_hours < self.max_age_hours

    def get(
        self,
        name: str,
        params: dict | None = None,
    ) -> pd.DataFrame | None:
        """
        Get cached data.

        Args:
            name: Cache entry name
            params: Optional parameters for unique key

        Returns:
            Cached DataFrame or None if not found/expired
        """
        if not self.is_cached(name, params):
            return None

        key = self._generate_key(name, params)
        cache_path = self._get_cache_path(key)

        try:
            df = pd.read_parquet(cache_path)
            logger.debug(f"Cache hit: {name} ({key})")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache {name}: {e}")
            return None

    def set(
        self,
        name: str,
        data: pd.DataFrame,
        params: dict | None = None,
    ) -> None:
        """
        Cache data.

        Args:
            name: Cache entry name
            data: DataFrame to cache
            params: Optional parameters for unique key
        """
        key = self._generate_key(name, params)
        cache_path = self._get_cache_path(key)

        try:
            data.to_parquet(cache_path)

            self._metadata[key] = {
                "name": name,
                "params": params,
                "created": datetime.now().isoformat(),
                "rows": len(data),
                "columns": list(data.columns),
            }
            self._save_metadata()

            logger.debug(f"Cached: {name} ({key}), {len(data)} rows")
        except Exception as e:
            logger.error(f"Failed to cache {name}: {e}")

    def delete(
        self,
        name: str,
        params: dict | None = None,
    ) -> bool:
        """
        Delete cached data.

        Args:
            name: Cache entry name
            params: Optional parameters for unique key

        Returns:
            True if deleted successfully
        """
        key = self._generate_key(name, params)
        cache_path = self._get_cache_path(key)

        try:
            if cache_path.exists():
                cache_path.unlink()
            if key in self._metadata:
                del self._metadata[key]
                self._save_metadata()
            logger.debug(f"Deleted cache: {name} ({key})")
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache {name}: {e}")
            return False

    def clear(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of entries cleared
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")

        self._metadata = {}
        self._save_metadata()

        logger.info(f"Cleared {count} cache entries")
        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        count = 0
        expired_keys = []

        for key, meta in self._metadata.items():
            cached_time = datetime.fromisoformat(meta["created"])
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600

            if age_hours >= self.max_age_hours:
                expired_keys.append(key)

        for key in expired_keys:
            cache_path = self._get_cache_path(key)
            try:
                if cache_path.exists():
                    cache_path.unlink()
                del self._metadata[key]
                count += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup {key}: {e}")

        if count > 0:
            self._save_metadata()
            logger.info(f"Cleaned up {count} expired cache entries")

        return count

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.parquet")
        )

        return {
            "entries": len(self._metadata),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
            "max_age_hours": self.max_age_hours,
        }

    def list_entries(self) -> pd.DataFrame:
        """
        List all cache entries.

        Returns:
            DataFrame with cache entry information
        """
        entries = []
        for key, meta in self._metadata.items():
            cache_path = self._get_cache_path(key)
            size_mb = cache_path.stat().st_size / (1024 * 1024) if cache_path.exists() else 0

            cached_time = datetime.fromisoformat(meta["created"])
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600

            entries.append(
                {
                    "key": key,
                    "name": meta["name"],
                    "rows": meta["rows"],
                    "columns": len(meta["columns"]),
                    "size_mb": size_mb,
                    "age_hours": age_hours,
                    "created": meta["created"],
                }
            )

        return pd.DataFrame(entries)


# Global cache instance
_cache = DataCache()


def cache_data(
    name: str,
    data: pd.DataFrame,
    params: dict | None = None,
) -> None:
    """
    Cache data using the global cache.

    Args:
        name: Cache entry name
        data: DataFrame to cache
        params: Optional parameters
    """
    _cache.set(name, data, params)


def load_cached_data(
    name: str,
    params: dict | None = None,
) -> pd.DataFrame | None:
    """
    Load data from global cache.

    Args:
        name: Cache entry name
        params: Optional parameters

    Returns:
        Cached DataFrame or None
    """
    return _cache.get(name, params)


def get_cache() -> DataCache:
    """Get the global cache instance."""
    return _cache
