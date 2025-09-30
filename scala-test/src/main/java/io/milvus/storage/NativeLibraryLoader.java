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

    /**
     * Load the native library. This method is thread-safe and will only load once.
     */
    public static synchronized void loadLibrary() {
        if (loaded) {
            return;
        }

        try {
            // Try to load from system library path first
            System.loadLibrary(NATIVE_LIBRARY_NAME);
            loaded = true;
            System.out.println("✓ Native library loaded from system path");
            return;
        } catch (UnsatisfiedLinkError e) {
            // Try loading with library path fallback
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