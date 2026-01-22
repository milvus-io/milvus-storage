"""
Properties wrapper for milvus-storage configuration.
"""

from typing import Dict, Optional

from ._ffi import check_result, get_ffi, get_library
from .exceptions import InvalidArgumentError


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
        if hasattr(self, "_props") and hasattr(self, "_lib"):
            self._lib.loon_properties_free(self._props)

    def __repr__(self) -> str:
        """String representation."""
        return f"Properties(count={self._props.count})"
