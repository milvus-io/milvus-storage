package io.milvus.storage;

/** A failed storage C API operation, retaining its Loon error code. */
public final class MilvusStorageException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    private final int errorCode;

    public MilvusStorageException(int errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public int errorCode() {
        return errorCode;
    }
}
