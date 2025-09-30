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
    private static Path tempLibraryPath = null;
    private static Path tempMainLibraryPath = null;

    /**
     * Load the native library. This method is thread-safe and will only load once.
     */
    public static synchronized void loadLibrary() {
        if (loaded) {
            return;
        }

        try {
            // Try to load from system library path first
            trySystemLoad();
            loaded = true;
            System.out.println("✓ Native library loaded from system path");
            return;
        } catch (UnsatisfiedLinkError e) {
            // Try loading with library path fallback
            System.out.println("System library not found, trying library path fallback...");
            if (tryLibraryPathLoad()) {
                loaded = true;
                System.out.println("✓ Native library loaded from library path");
                return;
            }
            System.out.println("Library path fallback failed, extracting from JAR...");
        }

        try {
            extractAndLoadFromJarWithPreload();
            loaded = true;
            System.out.println("✓ Native library loaded from JAR resources");
        } catch (Exception e) {
            throw new RuntimeException("Failed to load native library: " + e.getMessage(), e);
        }
    }

    private static void trySystemLoad() {
        System.loadLibrary(NATIVE_LIBRARY_NAME);
    }

    private static boolean tryLibraryPathLoad() {
        // Try to load both main library and JNI library from filesystem paths
        String[] basePaths = {
            "./scala-test/native/linux-x86_64/",
            "./native/linux-x86_64/",
            "native/linux-x86_64/",
            "./"
        };

        for (String basePath : basePaths) {
            try {
                // First try to load main library
                String mainLibPath = basePath + "libmilvus-storage.so";
                File mainLibFile = new File(mainLibPath);
                if (!mainLibFile.exists()) {
                    continue;
                }

                String jniLibPath = basePath + "lib" + NATIVE_LIBRARY_NAME + ".so";
                File jniLibFile = new File(jniLibPath);
                if (!jniLibFile.exists()) {
                    continue;
                }

                System.out.println("Trying to load libraries from: " + basePath);

                // Load main library first
                System.load(mainLibFile.getAbsolutePath());
                System.out.println("✓ Loaded main library: " + mainLibFile.getAbsolutePath());

                // Then load JNI library
                System.load(jniLibFile.getAbsolutePath());
                System.out.println("✓ Loaded JNI library: " + jniLibFile.getAbsolutePath());

                return true;
            } catch (UnsatisfiedLinkError e) {
                // Continue to next path
                System.out.println("Failed to load from " + basePath + ": " + e.getMessage());
            }
        }
        return false;
    }

    private static void extractAndLoadFromJarWithPreload() throws IOException {
        // First extract and load the main library dependency
        String mainLibraryPath = getMainLibraryPath();
        tempMainLibraryPath = extractLibraryFromJar(mainLibraryPath, "milvus_storage_");

        try {
            // Load main library first using System.load for now
            // Note: TLS issues can occur with static linking - consider dlopen if needed
            System.load(tempMainLibraryPath.toString());
            System.out.println("✓ Loaded main library: " + tempMainLibraryPath);
        } catch (UnsatisfiedLinkError e) {
            if (e.getMessage().contains("static TLS block")) {
                System.err.println("TLS allocation issue detected. This may be due to static linking.");
                System.err.println("Consider rebuilding libraries with -fPIC and dynamic linking flags.");
            }
            throw new RuntimeException("Failed to load main library: " + e.getMessage(), e);
        }

        // Extract the JNI library
        String jniLibraryPath = getNativeLibraryPath();
        tempLibraryPath = extractLibraryFromJar(jniLibraryPath, "milvus_storage_jni_");

        try {
            // Load JNI library - dependencies should now be resolved
            System.load(tempLibraryPath.toString());
            System.out.println("✓ Loaded JNI library: " + tempLibraryPath);

            // Register cleanup hook
            Runtime.getRuntime().addShutdownHook(new Thread(NativeLibraryLoader::cleanup));
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException("Failed to load JNI library: " + e.getMessage(), e);
        }
    }

    private static Path extractLibraryFromJar(String resourcePath, String prefix) throws IOException {
        try (InputStream is = NativeLibraryLoader.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new FileNotFoundException("Library not found in JAR: " + resourcePath);
            }

            String fileName = Paths.get(resourcePath).getFileName().toString();
            Path tempPath = Files.createTempFile(prefix, fileName);

            Files.copy(is, tempPath, StandardCopyOption.REPLACE_EXISTING);
            tempPath.toFile().setExecutable(true);

            return tempPath;
        }
    }

    private static String getNativeLibraryPath() {
        return getPlatformLibraryPath(getLibraryFileName());
    }

    private static String getMainLibraryPath() {
        return getPlatformLibraryPath(getMainLibraryFileName());
    }

    private static String getPlatformLibraryPath(String fileName) {
        String osName = System.getProperty("os.name").toLowerCase();
        String osArch = System.getProperty("os.arch").toLowerCase();

        String platformDir;
        if (osName.contains("linux")) {
            if (osArch.contains("aarch64") || osArch.contains("arm64")) {
                platformDir = "linux-aarch64";
            } else {
                platformDir = "linux-x86_64";
            }
        } else if (osName.contains("mac")) {
            if (osArch.contains("aarch64") || osArch.contains("arm64")) {
                platformDir = "darwin-aarch64";
            } else {
                platformDir = "darwin-x86_64";
            }
        } else if (osName.contains("win")) {
            platformDir = "win32-x86_64";
        } else {
            throw new UnsupportedOperationException("Unsupported platform: " + osName + "-" + osArch);
        }

        return "/native/" + platformDir + "/" + fileName;
    }

    private static String getLibraryFileName() {
        String osName = System.getProperty("os.name").toLowerCase();

        if (osName.contains("win")) {
            return NATIVE_LIBRARY_NAME + ".dll";
        } else if (osName.contains("mac")) {
            return "lib" + NATIVE_LIBRARY_NAME + ".dylib";
        } else {
            return "lib" + NATIVE_LIBRARY_NAME + ".so";
        }
    }

    private static String getMainLibraryFileName() {
        String osName = System.getProperty("os.name").toLowerCase();

        if (osName.contains("win")) {
            return "milvus-storage.dll";
        } else if (osName.contains("mac")) {
            return "libmilvus-storage.dylib";
        } else {
            return "libmilvus-storage.so";
        }
    }

    private static void cleanup() {
        if (tempLibraryPath != null) {
            try {
                Files.deleteIfExists(tempLibraryPath);
            } catch (IOException e) {
                // Ignore cleanup errors
                System.err.println("Warning: Failed to cleanup temporary JNI library file: " + e.getMessage());
            }
        }
        if (tempMainLibraryPath != null) {
            try {
                Files.deleteIfExists(tempMainLibraryPath);
            } catch (IOException e) {
                // Ignore cleanup errors
                System.err.println("Warning: Failed to cleanup temporary main library file: " + e.getMessage());
            }
        }
    }

    /**
     * Check if the native library is loaded.
     */
    public static boolean isLoaded() {
        return loaded;
    }
}