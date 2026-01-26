"""
Properties wrapper for milvus-storage configuration.
"""

from typing import Dict, Optional

from ._ffi import get_ffi, get_library

# Mapping from Python attribute names to C symbol names
_PROPERTY_KEY_MAPPING = {
    "FORMAT": "loon_properties_format",
    "FS_ADDRESS": "loon_properties_fs_address",
    "FS_BUCKET_NAME": "loon_properties_fs_bucket_name",
    "FS_ACCESS_KEY_ID": "loon_properties_fs_access_key_id",
    "FS_ACCESS_KEY_VALUE": "loon_properties_fs_access_key_value",
    "FS_ROOT_PATH": "loon_properties_fs_root_path",
    "FS_STORAGE_TYPE": "loon_properties_fs_storage_type",
    "FS_CLOUD_PROVIDER": "loon_properties_fs_cloud_provider",
    "FS_IAM_ENDPOINT": "loon_properties_fs_iam_endpoint",
    "FS_LOG_LEVEL": "loon_properties_fs_log_level",
    "FS_REGION": "loon_properties_fs_region",
    "FS_USE_SSL": "loon_properties_fs_use_ssl",
    "FS_SSL_CA_CERT": "loon_properties_fs_ssl_ca_cert",
    "FS_USE_IAM": "loon_properties_fs_use_iam",
    "FS_USE_VIRTUAL_HOST": "loon_properties_fs_use_virtual_host",
    "FS_REQUEST_TIMEOUT_MS": "loon_properties_fs_request_timeout_ms",
    "FS_GCP_NATIVE_WITHOUT_AUTH": "loon_properties_fs_gcp_native_without_auth",
    "FS_GCP_CREDENTIAL_JSON": "loon_properties_fs_gcp_credential_json",
    "FS_USE_CUSTOM_PART_UPLOAD": "loon_properties_fs_use_custom_part_upload",
    "FS_MAX_CONNECTIONS": "loon_properties_fs_max_connections",
    "FS_MULTI_PART_UPLOAD_SIZE": "loon_properties_fs_multi_part_upload_size",
    "WRITER_POLICY": "loon_properties_writer_policy",
    "WRITER_SCHEMA_BASE_PATTERNS": "loon_properties_writer_schema_base_patterns",
    "WRITER_SIZE_BASE_MACS": "loon_properties_writer_size_base_macs",
    "WRITER_SIZE_BASE_MCIG": "loon_properties_writer_size_base_mcig",
    "WRITER_BUFFER_SIZE": "loon_properties_writer_buffer_size",
    "WRITER_FILE_ROLLING_SIZE": "loon_properties_writer_file_rolling_size",
    "WRITER_COMPRESSION": "loon_properties_writer_compression",
    "WRITER_COMPRESSION_LEVEL": "loon_properties_writer_compression_level",
    "WRITER_ENABLE_DICTIONARY": "loon_properties_writer_enable_dictionary",
    "WRITER_ENC_ENABLE": "loon_properties_writer_enc_enable",
    "WRITER_ENC_KEY": "loon_properties_writer_enc_key",
    "WRITER_ENC_META": "loon_properties_writer_enc_meta",
    "WRITER_ENC_ALGORITHM": "loon_properties_writer_enc_algorithm",
    "WRITER_VORTEX_ENABLE_STATISTICS": "loon_properties_writer_vortex_enable_statistics",
    "READER_RECORD_BATCH_MAX_ROWS": "loon_properties_reader_record_batch_max_rows",
    "READER_RECORD_BATCH_MAX_SIZE": "loon_properties_reader_record_batch_max_size",
    "READER_LOGICAL_CHUNK_ROWS": "loon_properties_reader_logical_chunk_rows",
    "TRANSACTION_COMMIT_NUM_RETRIES": "loon_properties_transaction_commit_num_retries",
}


class _PropertyKeysMeta(type):
    """Metaclass for PropertyKeys to enable class-level property access."""

    _cache: Dict[str, str] = {}

    def __getattr__(cls, name: str) -> str:
        if name.startswith("_"):
            raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

        c_name = _PROPERTY_KEY_MAPPING.get(name)
        if c_name is None:
            raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

        if c_name not in cls._cache:
            lib = get_library().lib
            ffi = get_ffi()
            c_str = getattr(lib, c_name)
            cls._cache[c_name] = ffi.string(c_str).decode("utf-8")

        return cls._cache[c_name]


class PropertyKeys(metaclass=_PropertyKeysMeta):
    """
    Property key constants from C library.

    These are loaded lazily from the C library to ensure they stay in sync.

    Example:
        >>> from milvus_storage import PropertyKeys
        >>> props = {
        ...     PropertyKeys.FS_STORAGE_TYPE: "local",
        ...     PropertyKeys.FS_ROOT_PATH: "/tmp",
        ... }
    """

    pass


from ._ffi import check_result  # noqa: E402
from .exceptions import InvalidArgumentError  # noqa: E402


class Properties:
    """
    Configuration properties for milvus-storage.

    Properties can be used to configure both Writer and Reader behavior.

    Common properties:
        - storage.memory.limit: Memory limit in bytes
        - storage.row_group.max_size: Max row group size
        - storage.batch.size: Batch size for reading
        - storage.aws.access_key_id: AWS access key
        - storage.aws.secret_access_key: AWS secret key
        - storage.aws.region: AWS region
        - storage.azure.account_name: Azure account name
        - storage.azure.account_key: Azure account key

    Example:
        >>> props = Properties({
        ...     "storage.memory.limit": "1073741824",  # 1GB
        ...     "storage.row_group.max_size": "1048576"  # 1MB
        ... })
    """

    def __init__(self, properties: Optional[Dict[str, str]] = None):
        """
        Initialize properties.

        Args:
            properties: Dictionary of property key-value pairs.
                       Both keys and values must be strings.
        """
        self._ffi = get_ffi()
        self._lib = get_library().lib
        self._props = self._ffi.new("struct LoonProperties*")

        if properties:
            self._create_from_dict(properties)
        else:
            # Create empty properties
            self._props.properties = self._ffi.NULL
            self._props.count = 0

    def _create_from_dict(self, properties: Dict[str, str]):
        """Create C properties from Python dict."""
        if not properties:
            self._props.properties = self._ffi.NULL
            self._props.count = 0
            return

        # Validate all values are strings
        for key, value in properties.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise InvalidArgumentError(
                    f"Property keys and values must be strings, "
                    f"got {type(key).__name__}: {type(value).__name__}"
                )

        # Convert to C arrays
        # cffi requires char* to be created individually
        keys_list = list(properties.keys())
        values_list = list(properties.values())

        # Create cffi char* objects for each string
        keys_c = [self._ffi.new("char[]", k.encode("utf-8")) for k in keys_list]
        values_c = [self._ffi.new("char[]", v.encode("utf-8")) for v in values_list]

        # Create cffi char** arrays
        keys_array = self._ffi.new("char*[]", keys_c)
        values_array = self._ffi.new("char*[]", values_c)

        # Call C API
        result = self._lib.loon_properties_create(
            keys_array, values_array, len(keys_list), self._props
        )
        check_result(result)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a property value by key.

        Args:
            key: Property key to look up
            default: Default value if key not found

        Returns:
            Property value or default if not found
        """
        key_bytes = key.encode("utf-8")
        value = self._lib.loon_properties_get(self._props, key_bytes)

        if value != self._ffi.NULL:
            return self._ffi.string(value).decode("utf-8")
        return default

    def _get_c_properties(self):
        """Get pointer to C properties structure."""
        return self._props

    def __del__(self):
        """Clean up C resources."""
        self._lib.loon_properties_free(self._props)

    def __repr__(self) -> str:
        """String representation."""
        return f"Properties(count={self._props.count})"
