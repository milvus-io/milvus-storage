package io.milvus.storage;

import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;
import java.util.jar.*;
import java.net.URISyntaxException;

/**
 * Native library loader that extracts and loads native libraries from JAR resources.
 * Supports multiple platforms and automatic cleanup.
 */
public class NativeLibraryLoader {
    private static boolean loaded = false;
    private static final String JNI_LIBRARY_NAME = "milvus-storage-jni";

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
     * Load libraries from JAR resources.
     * Extracts ALL native .so files to a temp directory, then loads the JNI library.
     * The JNI .so has RUNPATH=$ORIGIN so it finds dependencies in the same directory.
     */
    private static void loadFromResources() throws IOException {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        String libExtension;
        String platform;

        if (os.contains("windows")) {
            libExtension = "dll";
            platform = "windows-" + (arch.contains("64") ? "x86_64" : "x86");
        } else if (os.contains("mac")) {
            libExtension = "dylib";
            platform = "darwin-" + (arch.contains("aarch64") || arch.contains("arm64") ? "aarch64" : "x86_64");
        } else {
            libExtension = "so";
            platform = "linux-" + (arch.contains("aarch64") || arch.contains("arm64") ? "aarch64" : "x86_64");
        }

        String jniLibName = "lib" + JNI_LIBRARY_NAME + "." + libExtension;

        // Create temp directory for all native libraries
        File tempDir = new File(System.getProperty("java.io.tmpdir"), "milvus-storage-native");
        tempDir.mkdirs();

        // Extract all native libraries from JAR resources
        extractAllNativeLibs(tempDir);

        // Load the JNI library (RUNPATH=$ORIGIN will find deps in same dir)
        File jniLib = new File(tempDir, jniLibName);
        if (!jniLib.exists()) {
            throw new IOException("JNI library not found after extraction: " + jniLibName);
        }
        System.load(jniLib.getAbsolutePath());
    }

    /**
     * Extract all native library files from the JAR's /native/ directory to tempDir.
     * Scans JAR entries for paths starting with "native/" and ending with
     * .so, .dylib, or .dll.
     */
    private static void extractAllNativeLibs(File tempDir) throws IOException {
        String nativePrefix = "native/";
        URL url = NativeLibraryLoader.class.getResource("/" + nativePrefix);
        if (url == null) {
            throw new IOException("Native resource directory not found in JAR");
        }

        String protocol = url.getProtocol();
        if ("jar".equals(protocol)) {
            extractFromJar(tempDir, nativePrefix);
        } else if ("file".equals(protocol)) {
            // Running from filesystem (e.g. during development/testing)
            extractFromFileSystem(tempDir, nativePrefix, url);
        } else {
            throw new IOException("Unsupported resource protocol: " + protocol);
        }
    }

    /**
     * Extract native libraries from a JAR file.
     */
    private static void extractFromJar(File tempDir, String nativePrefix) throws IOException {
        URL jarUrl = NativeLibraryLoader.class.getProtectionDomain().getCodeSource().getLocation();
        String jarPath;
        try {
            jarPath = jarUrl.toURI().getPath();
        } catch (URISyntaxException e) {
            throw new IOException("Invalid JAR path: " + jarUrl, e);
        }

        int extracted = 0;
        try (JarFile jar = new JarFile(jarPath)) {
            Enumeration<JarEntry> entries = jar.entries();
            while (entries.hasMoreElements()) {
                JarEntry entry = entries.nextElement();
                String name = entry.getName();

                if (!name.startsWith(nativePrefix) || entry.isDirectory()) {
                    continue;
                }
                if (!isNativeLibrary(name)) {
                    continue;
                }

                // Flatten: strip the native/<platform>/ prefix, keep subdirs like ossl-modules/
                String relativePath = stripPlatformPrefix(name, nativePrefix);
                if (relativePath == null) {
                    continue;
                }

                File outFile = new File(tempDir, relativePath);
                outFile.getParentFile().mkdirs();

                try (InputStream is = jar.getInputStream(entry)) {
                    Files.copy(is, outFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                }
                outFile.deleteOnExit();
                extracted++;
            }
        }
        System.out.println("Extracted " + extracted + " native libraries from JAR");
    }

    /**
     * Extract native libraries from the filesystem (development mode).
     */
    private static void extractFromFileSystem(File tempDir, String nativePrefix, URL baseUrl) throws IOException {
        File nativeDir;
        try {
            nativeDir = new File(baseUrl.toURI());
        } catch (URISyntaxException e) {
            throw new IOException("Invalid native dir path: " + baseUrl, e);
        }

        if (!nativeDir.isDirectory()) {
            throw new IOException("Native resource path is not a directory: " + nativeDir);
        }

        int extracted = 0;
        // Walk all platform subdirectories
        File[] platformDirs = nativeDir.listFiles(File::isDirectory);
        if (platformDirs == null) {
            return;
        }

        for (File platformDir : platformDirs) {
            extracted += extractFromDirectory(platformDir, tempDir, "");
        }
        System.out.println("Extracted " + extracted + " native libraries from filesystem");
    }

    /**
     * Recursively extract native libraries from a directory.
     */
    private static int extractFromDirectory(File dir, File tempDir, String relativePath) throws IOException {
        int extracted = 0;
        File[] files = dir.listFiles();
        if (files == null) {
            return 0;
        }

        for (File file : files) {
            String childPath = relativePath.isEmpty() ? file.getName() : relativePath + "/" + file.getName();
            if (file.isDirectory()) {
                extracted += extractFromDirectory(file, tempDir, childPath);
            } else if (isNativeLibrary(file.getName())) {
                File outFile = new File(tempDir, childPath);
                outFile.getParentFile().mkdirs();
                Files.copy(file.toPath(), outFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                outFile.deleteOnExit();
                extracted++;
            }
        }
        return extracted;
    }

    /**
     * Strip the native/<platform>/ prefix from a JAR entry name, returning the
     * relative path under the platform directory (e.g. "libfoo.so" or "ossl-modules/bar.so").
     * Returns null if the entry doesn't match expected structure.
     */
    private static String stripPlatformPrefix(String entryName, String nativePrefix) {
        // entryName looks like: native/linux-x86_64/libfoo.so
        //                    or: native/linux-x86_64/ossl-modules/bar.so
        String afterNative = entryName.substring(nativePrefix.length());
        int slashIdx = afterNative.indexOf('/');
        if (slashIdx < 0) {
            return null;
        }
        String relativePath = afterNative.substring(slashIdx + 1);
        return relativePath.isEmpty() ? null : relativePath;
    }

    /**
     * Check if a filename looks like a native library.
     */
    private static boolean isNativeLibrary(String name) {
        return name.endsWith(".so") || name.endsWith(".dylib") || name.endsWith(".dll");
    }

    /**
     * Check if the native library is loaded.
     */
    public static boolean isLoaded() {
        return loaded;
    }
}
