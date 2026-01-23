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
     * Load libraries from JAR resources
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

        // Create temp directory for extracted libraries
        File tempDir = new File(System.getProperty("java.io.tmpdir"), "milvus-storage-native");
        tempDir.mkdirs();

        // Extract all libraries from the platform directory
        // This ensures dependencies are available when loading the JNI library
        extractAllLibraries("/" + platform + "/", tempDir, libExtension);

        // Load the JNI library (dependencies will be found via RPATH)
        String jniLibName = "lib" + JNI_LIBRARY_NAME + "." + libExtension;
        File jniLib = new File(tempDir, jniLibName);
        if (!jniLib.exists()) {
            throw new IOException("Could not find native library: " + jniLib.getAbsolutePath());
        }
        System.load(jniLib.getAbsolutePath());
    }

    /**
     * Extract all libraries from a resource directory to a temp directory.
     * Dynamically discovers all files in the JAR's platform directory.
     */
    private static void extractAllLibraries(String resourceDir, File destDir, String libExtension) throws IOException {
        // Get the JAR file containing this class
        URL classUrl = NativeLibraryLoader.class.getResource("/" + NativeLibraryLoader.class.getName().replace('.', '/') + ".class");
        if (classUrl == null) {
            throw new IOException("Cannot find class resource");
        }

        String protocol = classUrl.getProtocol();
        // Remove leading slash from resourceDir for JAR entry matching
        String dirPath = resourceDir.startsWith("/") ? resourceDir.substring(1) : resourceDir;

        if ("jar".equals(protocol)) {
            // Running from JAR - enumerate all entries
            String jarPath = classUrl.getPath().substring(5, classUrl.getPath().indexOf("!"));
            try (JarFile jar = new JarFile(URLDecoder.decode(jarPath, "UTF-8"))) {
                Enumeration<JarEntry> entries = jar.entries();
                while (entries.hasMoreElements()) {
                    JarEntry entry = entries.nextElement();
                    String name = entry.getName();
                    // Check if entry is in our platform directory and is a library file
                    if (name.startsWith(dirPath) && !entry.isDirectory()) {
                        String fileName = name.substring(dirPath.length());
                        if (fileName.contains("." + libExtension)) {
                            try {
                                extractLibraryFromResource("/" + name, fileName, destDir);
                            } catch (IOException e) {
                                // Continue on extraction failure
                            }
                        }
                    }
                }
            }
        } else {
            // Running from file system (e.g., during development)
            URL dirUrl = NativeLibraryLoader.class.getResource(resourceDir);
            if (dirUrl != null && "file".equals(dirUrl.getProtocol())) {
                try {
                    File dir = new File(dirUrl.toURI());
                    if (dir.isDirectory()) {
                        for (File file : dir.listFiles()) {
                            if (file.getName().contains("." + libExtension)) {
                                Files.copy(file.toPath(), new File(destDir, file.getName()).toPath(),
                                    StandardCopyOption.REPLACE_EXISTING);
                            }
                        }
                    }
                } catch (URISyntaxException e) {
                    throw new IOException("Invalid URI: " + dirUrl, e);
                }
            }
        }
    }

    /**
     * Extract library from JAR resources to a temporary file
     */
    private static File extractLibraryFromResource(String resourcePath, String libName, File destDir) throws IOException {
        InputStream resourceStream = NativeLibraryLoader.class.getResourceAsStream(resourcePath);

        if (resourceStream == null) {
            throw new IOException("Resource not found: " + resourcePath);
        }

        try {
            File tempFile = new File(destDir, libName);

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