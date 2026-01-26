"""Tests for PropertyKeys class."""

from milvus_storage import PropertyKeys


class TestPropertyKeys:
    """Test PropertyKeys loads constants from C library."""

    def test_fs_storage_type(self):
        """Test FS_STORAGE_TYPE property key."""
        key = PropertyKeys.FS_STORAGE_TYPE
        assert isinstance(key, str)
        assert len(key) > 0
        assert "storage" in key.lower() or "fs" in key.lower()

    def test_fs_root_path(self):
        """Test FS_ROOT_PATH property key."""
        key = PropertyKeys.FS_ROOT_PATH
        assert isinstance(key, str)
        assert len(key) > 0

    def test_writer_buffer_size(self):
        """Test WRITER_BUFFER_SIZE property key."""
        key = PropertyKeys.WRITER_BUFFER_SIZE
        assert isinstance(key, str)
        assert len(key) > 0

    def test_reader_record_batch_max_rows(self):
        """Test READER_RECORD_BATCH_MAX_ROWS property key."""
        key = PropertyKeys.READER_RECORD_BATCH_MAX_ROWS
        assert isinstance(key, str)
        assert len(key) > 0

    def test_format(self):
        """Test FORMAT property key."""
        key = PropertyKeys.FORMAT
        assert isinstance(key, str)
        assert len(key) > 0

    def test_caching(self):
        """Test that property keys are cached."""
        # Access twice, should use cache
        key1 = PropertyKeys.FS_STORAGE_TYPE
        key2 = PropertyKeys.FS_STORAGE_TYPE
        assert key1 == key2
        # Cache is stored in the metaclass
        assert "loon_properties_fs_storage_type" in type(PropertyKeys)._cache

    def test_all_fs_keys(self):
        """Test all FS property keys are accessible."""
        keys = [
            PropertyKeys.FS_ADDRESS,
            PropertyKeys.FS_BUCKET_NAME,
            PropertyKeys.FS_ACCESS_KEY_ID,
            PropertyKeys.FS_ACCESS_KEY_VALUE,
            PropertyKeys.FS_ROOT_PATH,
            PropertyKeys.FS_STORAGE_TYPE,
            PropertyKeys.FS_CLOUD_PROVIDER,
            PropertyKeys.FS_IAM_ENDPOINT,
            PropertyKeys.FS_LOG_LEVEL,
            PropertyKeys.FS_REGION,
            PropertyKeys.FS_USE_SSL,
            PropertyKeys.FS_SSL_CA_CERT,
            PropertyKeys.FS_USE_IAM,
            PropertyKeys.FS_USE_VIRTUAL_HOST,
            PropertyKeys.FS_REQUEST_TIMEOUT_MS,
            PropertyKeys.FS_GCP_NATIVE_WITHOUT_AUTH,
            PropertyKeys.FS_GCP_CREDENTIAL_JSON,
            PropertyKeys.FS_USE_CUSTOM_PART_UPLOAD,
            PropertyKeys.FS_MAX_CONNECTIONS,
            PropertyKeys.FS_MULTI_PART_UPLOAD_SIZE,
        ]
        for key in keys:
            assert isinstance(key, str)
            assert len(key) > 0

    def test_all_writer_keys(self):
        """Test all Writer property keys are accessible."""
        keys = [
            PropertyKeys.WRITER_POLICY,
            PropertyKeys.WRITER_SCHEMA_BASE_PATTERNS,
            PropertyKeys.WRITER_SIZE_BASE_MACS,
            PropertyKeys.WRITER_SIZE_BASE_MCIG,
            PropertyKeys.WRITER_BUFFER_SIZE,
            PropertyKeys.WRITER_FILE_ROLLING_SIZE,
            PropertyKeys.WRITER_COMPRESSION,
            PropertyKeys.WRITER_COMPRESSION_LEVEL,
            PropertyKeys.WRITER_ENABLE_DICTIONARY,
            PropertyKeys.WRITER_ENC_ENABLE,
            PropertyKeys.WRITER_ENC_KEY,
            PropertyKeys.WRITER_ENC_META,
            PropertyKeys.WRITER_ENC_ALGORITHM,
            PropertyKeys.WRITER_VORTEX_ENABLE_STATISTICS,
        ]
        for key in keys:
            assert isinstance(key, str)
            assert len(key) > 0

    def test_all_reader_keys(self):
        """Test all Reader property keys are accessible."""
        keys = [
            PropertyKeys.READER_RECORD_BATCH_MAX_ROWS,
            PropertyKeys.READER_RECORD_BATCH_MAX_SIZE,
            PropertyKeys.READER_LOGICAL_CHUNK_ROWS,
        ]
        for key in keys:
            assert isinstance(key, str)
            assert len(key) > 0

    def test_transaction_keys(self):
        """Test Transaction property keys are accessible."""
        key = PropertyKeys.TRANSACTION_COMMIT_NUM_RETRIES
        assert isinstance(key, str)
        assert len(key) > 0
