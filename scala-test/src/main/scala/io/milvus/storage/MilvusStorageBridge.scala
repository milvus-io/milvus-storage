package io.milvus.storage

class MilvusStorageBridge {
    // Ensure native library is loaded
    NativeLibraryLoader.loadLibrary()

    @native def CallTheCFunction(): Long
}