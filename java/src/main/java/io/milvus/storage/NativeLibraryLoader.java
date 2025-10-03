package io.milvus.storage;

import java.io.*;
import java.nio.file.*;

/**
 * Native library loader that extracts and loads native libraries from JAR resources.
 * Supports multiple platforms and automatic cleanup.
 */
public class NativeLibraryLoader {
    private static boolean loaded = false;
    private static final String JNI_LIBRARY_NAME = "milvus_storage_jni";
    private static final String MILVUS_LIBRARY_NAME = "milvus-storage";

    /**
     * Load the native library. This method is thread-safe and will only load once.
     */
    public static synchronized void loadLibrary() {
        if (loaded) {
            return;
        }

        try {
            // Try loading from resources first (for fat jar)
            loadFromResources();
            loaded = true;
            System.out.println("Native libraries loaded from resources");
            return;
        } catch (Exception e) {
            System.out.println("Could not load from resources: " + e.getMessage());
        }

        try {
            // Fall back to system library path
            System.loadLibrary(JNI_LIBRARY_NAME);
            loaded = true;
            System.out.println("Native library loaded from system path");
            return;
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException(
                "Failed to load native library '" + JNI_LIBRARY_NAME + "'. " +
                "Please ensure the library is in java.library.path or packaged in the JAR.", e);
        }
    }

    /**
     * Load libraries from JAR resources
     */
    private static void loadFromResources() throws IOException {
        String os = System.getProperty("os.name").toLowerCase();
        String libExtension;

        if (os.contains("windows")) {
            libExtension = "dll";
        } else if (os.contains("mac")) {
            libExtension = "dylib";
        } else {
            libExtension = "so";
        }

        String jniLibName = "lib" + JNI_LIBRARY_NAME + "." + libExtension;
        String milvusLibName = "lib" + MILVUS_LIBRARY_NAME + "." + libExtension;

        // Load milvus-storage first (dependency)
        File milvusLib = extractLibraryFromResource("/native/" + milvusLibName, milvusLibName);
        if (milvusLib != null && milvusLib.exists()) {
            System.load(milvusLib.getAbsolutePath());
        }

        // Load JNI library
        File jniLib = extractLibraryFromResource("/native/" + jniLibName, jniLibName);
        if (jniLib == null || !jniLib.exists()) {
            throw new IOException("Could not find native library in resources: /native/" + jniLibName);
        }
        System.load(jniLib.getAbsolutePath());
    }

    /**
     * Extract library from JAR resources to a temporary file
     */
    private static File extractLibraryFromResource(String resourcePath, String libName) throws IOException {
        InputStream resourceStream = NativeLibraryLoader.class.getResourceAsStream(resourcePath);

        if (resourceStream == null) {
            return null;
        }

        try {
            // Create temp directory
            File tempDir = new File(System.getProperty("java.io.tmpdir"), "milvus-storage-native");
            tempDir.mkdirs();

            File tempFile = new File(tempDir, libName);

            // Copy resource to temp file
            Files.copy(resourceStream, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            tempFile.deleteOnExit();

            return tempFile;
        } finally {
            resourceStream.close();
        }
    }

    /**
     * Check if the native library is loaded.
     */
    public static boolean isLoaded() {
        return loaded;
    }
}