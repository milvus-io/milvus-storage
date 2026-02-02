"""
Configuration loader for milvus-storage integration tests.

Loads storage backend configuration from YAML files and provides
utilities to convert to FFI Properties format.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from milvus_storage import PropertyKeys


# Default config file paths
CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG = CONFIG_DIR / "config.yaml"
LOCAL_CONFIG = CONFIG_DIR / "config.local.yaml"


def _expand_env_vars(value: Any) -> Any:
    """Expand environment variables in string values.

    Supports ${VAR_NAME} syntax. Returns original value if not a string
    or if the environment variable is not set.
    """
    if not isinstance(value, str):
        return value

    pattern = re.compile(r'\$\{([^}]+)\}')

    def replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return pattern.sub(replace, value)


def _expand_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand environment variables in config dict."""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _expand_config(value)
        elif isinstance(value, list):
            result[key] = [_expand_env_vars(item) for item in value]
        else:
            result[key] = _expand_env_vars(value)
    return result


class TestConfig:
    """Test configuration manager.

    Configuration priority:
    1. TEST_CONFIG_FILE environment variable
    2. tests/config.local.yaml (gitignored)
    3. tests/config.yaml (default)
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Load configuration from file.

        Args:
            config_path: Optional path to config file. If not provided,
                        uses the priority order described above.
        """
        self._config_path = self._resolve_config_path(config_path)
        self._config = self._load_config()

    def _resolve_config_path(self, config_path: Optional[Path]) -> Path:
        """Resolve which config file to use."""
        # 1. Explicit path
        if config_path:
            return Path(config_path)

        # 2. Environment variable
        env_config = os.environ.get("TEST_CONFIG_FILE")
        if env_config:
            return Path(env_config)

        # 3. Local config (gitignored)
        if LOCAL_CONFIG.exists():
            return LOCAL_CONFIG

        # 4. Default config
        return DEFAULT_CONFIG

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse configuration file."""
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self._config_path}")

        with open(self._config_path, "r") as f:
            config = yaml.safe_load(f)

        return _expand_config(config)

    @property
    def config_path(self) -> Path:
        """Return the path to the loaded config file."""
        return self._config_path

    @property
    def format(self) -> str:
        """Return the data format (parquet or vortex)."""
        return self._config.get("format", "parquet")

    @property
    def storage_backend(self) -> str:
        """Return the active storage backend name."""
        return self._config.get("storage_backend", "local")

    @property
    def backend_config(self) -> Dict[str, Any]:
        """Return configuration for the active storage backend."""
        backend = self.storage_backend
        return self._config.get(backend, {})

    @property
    def is_local(self) -> bool:
        """Check if using local filesystem backend."""
        return self.storage_backend == "local"

    @property
    def is_cloud(self) -> bool:
        """Check if using a cloud storage backend."""
        return self.storage_backend in (
            "aws", "gcs", "azure", "aliyun", "tencent", "huawei"
        )

    @property
    def is_s3_compatible(self) -> bool:
        """Check if using an S3-compatible backend."""
        return self.storage_backend in (
            "minio", "aws", "aliyun", "tencent", "huawei"
        )

    @property
    def root_path(self) -> str:
        """Get the root path for local filesystem (SubtreeFilesystem root)."""
        if self.is_local:
            return self.backend_config.get("root_path", "/tmp/milvus-storage-test")
        return ""

    @property
    def bucket_name(self) -> str:
        """Get the bucket name for cloud storage (SubtreeFilesystem root)."""
        return self.backend_config.get("bucket_name", "")

    @property
    def base_path(self) -> str:
        """Get the base path (relative to SubtreeFilesystem root)."""
        return self._config.get("base_path", "")

    def get_base_path(self) -> str:
        """Get the base path for storage (relative to SubtreeFilesystem root)."""
        return self.base_path

    def to_fs_properties(self) -> Dict[str, str]:
        """Convert backend config to filesystem properties dict.

        Returns a dict that can be passed to Properties for FFI calls.
        """
        backend = self.storage_backend
        config = self.backend_config

        if backend == "local":
            # Local filesystem uses root_path
            props = {}
            if self.root_path:
                props[PropertyKeys.FS_ROOT_PATH] = self.root_path
            return props

        # Remote storage - all backends use the same property names
        props = {}

        if "cloud_provider" in config:
            props[PropertyKeys.FS_CLOUD_PROVIDER] = config["cloud_provider"]
        if "address" in config:
            props[PropertyKeys.FS_ADDRESS] = config["address"]
        if self.bucket_name:
            props[PropertyKeys.FS_BUCKET_NAME] = self.bucket_name
        if "access_key" in config:
            props[PropertyKeys.FS_ACCESS_KEY_ID] = config["access_key"]
        if "secret_key" in config:
            props[PropertyKeys.FS_ACCESS_KEY_VALUE] = config["secret_key"]
        if "region" in config:
            props[PropertyKeys.FS_REGION] = config["region"]

        return props

    def get_writer_properties(
        self,
        file_rolling_size: Optional[int] = None,
        buffer_size: Optional[int] = None,
        compression: Optional[str] = None,
        **extra_props
    ) -> Dict[str, str]:
        """Get writer properties including filesystem config.

        Args:
            file_rolling_size: File rolling size in bytes
            buffer_size: Write buffer size in bytes
            compression: Compression codec (snappy, gzip, zstd, lz4, none)
            **extra_props: Additional properties to include

        Returns:
            Dict of properties for Writer creation
        """
        props = self.to_fs_properties()
        props[PropertyKeys.FORMAT] = self.format

        if file_rolling_size is not None:
            props[PropertyKeys.WRITER_FILE_ROLLING_SIZE] = str(file_rolling_size)
        if buffer_size is not None:
            props[PropertyKeys.WRITER_BUFFER_SIZE] = str(buffer_size)
        if compression is not None:
            props[PropertyKeys.WRITER_COMPRESSION] = compression

        props.update({k: str(v) for k, v in extra_props.items()})
        return props

    def get_reader_properties(self, **extra_props) -> Dict[str, str]:
        """Get reader properties including filesystem config.

        Args:
            **extra_props: Additional properties to include

        Returns:
            Dict of properties for Reader creation
        """
        props = self.to_fs_properties()
        props.update({k: str(v) for k, v in extra_props.items()})
        return props

    def get_transaction_properties(
        self,
        num_retries: int = 3,
        **extra_props
    ) -> Dict[str, str]:
        """Get transaction properties including filesystem config.

        Args:
            num_retries: Number of commit retries on conflict
            **extra_props: Additional properties to include

        Returns:
            Dict of properties for Transaction creation
        """
        props = self.to_fs_properties()
        props[PropertyKeys.TRANSACTION_COMMIT_NUM_RETRIES] = str(num_retries)
        props.update({k: str(v) for k, v in extra_props.items()})
        return props


# Global config instance (lazy loaded)
_config: Optional[TestConfig] = None


def get_config() -> TestConfig:
    """Get the global test configuration instance."""
    global _config
    if _config is None:
        _config = TestConfig()
    return _config


def reload_config(config_path: Optional[Path] = None) -> TestConfig:
    """Reload configuration, optionally from a different path."""
    global _config
    _config = TestConfig(config_path)
    return _config
