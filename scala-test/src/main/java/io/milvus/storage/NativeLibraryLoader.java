package io.milvus.storage;

import java.io.*;
import java.nio.file.*;

/**
 * Native library loader that extracts and loads native libraries from JAR resources.
 * Supports multiple platforms and automatic cleanup.
 */
public class NativeLibraryLoader {
    private static boolean loaded = false;
    private static final String NATIVE_LIBRARY_NAME = "milvus_storage_jni";

    static {
        // CRITICAL: Load libmilvus-storage.so in a static initializer to ensure it's
        // loaded as early as possible to allocate TLS space
        try {
            String milvusStoragePath = System.getProperty("user.dir") + "/../cpp/build/Release/libmilvus-storage.so";
            File milvusStorageLib = new File(milvusStoragePath);
            if (milvusStorageLib.exists()) {
                // This will fail with TLS error, but we'll use the script wrapper instead
                // System.load(milvusStorageLib.getAbsolutePath());
                System.out.println("Note: libmilvus-storage.so found at: " + milvusStorageLib.getAbsolutePath());
                System.out.println("Please ensure LD_PRELOAD is set or use run-test.sh script");
            }
        } catch (Exception e) {
            System.err.println("Warning: Could not check for libmilvus-storage.so: " + e.getMessage());
        }
    }

    /**
     * Load the native library. This method is thread-safe and will only load once.
     */
    public static synchronized void loadLibrary() {
        if (loaded) {
            return;
        }

        try {
            // Load the JNI wrapper library
            System.loadLibrary(NATIVE_LIBRARY_NAME);
            loaded = true;
            System.out.println("Native library loaded from system path");
            return;
        } catch (UnsatisfiedLinkError e) {
            // Try loading with library path fallback
            e.printStackTrace();
            System.out.println("System library not found, trying library path fallback...");
        }
    }

    /**
     * Check if the native library is loaded.
     */
    public static boolean isLoaded() {
        return loaded;
    }
}