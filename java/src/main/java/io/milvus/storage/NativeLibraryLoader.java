package io.milvus.storage;

import java.io.IOException;
import java.io.InputStream;
import java.net.JarURLConnection;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Enumeration;
import java.util.Locale;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Stream;

/** Loads JNI from an explicit path, the current platform's resources, or the system path. */
public class NativeLibraryLoader {
    private static final String JNI_LIBRARY_NAME = "milvus-storage-jni";
    private static final String NATIVE_PATH = "milvus.storage.native.path";
    private static volatile boolean loaded;

    public NativeLibraryLoader() {}

    /**
     * Uses {@code milvus.storage.native.path}, when set, before packaged resources.
     * The property must name an existing absolute JNI library path. Explicit path,
     * extraction and linking failures remain visible; the system path is used only
     * when neither an explicit path nor packaged JNI exists.
     */
    public static synchronized void loadLibrary() {
        if (loaded) {
            return;
        }
        Path explicit = explicitLibrary();
        if (explicit != null) {
            System.load(explicit.toString());
            loaded = true;
            return;
        }
        String platform = currentPlatform();
        String prefix = "native/" + platform + "/";
        String libraryName = System.mapLibraryName(JNI_LIBRARY_NAME);
        URL resource = NativeLibraryLoader.class.getResource("/" + prefix + libraryName);
        if (resource == null) {
            System.loadLibrary(JNI_LIBRARY_NAME);
        } else {
            try {
                Path directory = extractLibraries(resource, prefix);
                Path storage = directory.resolve(System.mapLibraryName("milvus-storage"));
                if (Files.isRegularFile(storage)) {
                    System.load(storage.toAbsolutePath().toString());
                }
                System.load(directory.resolve(libraryName).toAbsolutePath().toString());
            } catch (IOException error) {
                UnsatisfiedLinkError failure = new UnsatisfiedLinkError(
                        "Cannot extract storage JNI for " + platform + ": " + error.getMessage());
                failure.initCause(error);
                throw failure;
            }
        }
        loaded = true;
    }

    static Path explicitLibrary() {
        String configured = System.getProperty(NATIVE_PATH);
        if (configured == null) {
            return null;
        }
        Path path = Paths.get(configured);
        if (!path.isAbsolute() || !Files.isRegularFile(path)) {
            throw new UnsatisfiedLinkError(
                    NATIVE_PATH + " must name an existing absolute library path");
        }
        return path;
    }

    static String currentPlatform() {
        String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
        String platform;
        if (os.contains("linux")) {
            platform = "linux";
        } else if (os.contains("mac") || os.contains("darwin")) {
            platform = "darwin";
        } else if (os.contains("windows")) {
            platform = "windows";
        } else {
            throw new UnsatisfiedLinkError("Unsupported operating system: " + os);
        }
        if (arch.equals("amd64") || arch.equals("x86_64")) {
            arch = "x86_64";
        } else if (arch.equals("aarch64") || arch.equals("arm64")) {
            arch = "aarch64";
        } else if (arch.equals("x86") || arch.matches("i[3-6]86")) {
            arch = "x86";
        } else {
            throw new UnsatisfiedLinkError("Unsupported architecture: " + arch);
        }
        return platform + "-" + arch;
    }

    // The resource URL identifies the artifact containing the native libraries,
    // which may differ from the artifact containing this Java API class.
    static Path extractLibraries(URL jniResource, String prefix) throws IOException {
        Path directory = Files.createTempDirectory("milvus-storage-native-");
        directory.toFile().deleteOnExit();
        if ("jar".equals(jniResource.getProtocol())) {
            JarURLConnection connection = (JarURLConnection) jniResource.openConnection();
            connection.setUseCaches(false);
            try (JarFile jar = connection.getJarFile()) {
                Enumeration<JarEntry> entries = jar.entries();
                while (entries.hasMoreElements()) {
                    JarEntry entry = entries.nextElement();
                    String name = entry.getName();
                    if (entry.isDirectory() || !name.startsWith(prefix) || !isNativeLibrary(name)) {
                        continue;
                    }
                    Path destination = destination(directory, name.substring(prefix.length()));
                    try (InputStream input = jar.getInputStream(entry)) {
                        Files.copy(input, destination);
                    }
                    destination.toFile().deleteOnExit();
                }
            }
        } else if ("file".equals(jniResource.getProtocol())) {
            final Path source;
            try {
                source = Paths.get(jniResource.toURI()).getParent();
            } catch (URISyntaxException error) {
                throw new IOException("Invalid storage resource URL", error);
            }
            try (Stream<Path> paths = Files.walk(source)) {
                Path[] files = paths.filter(Files::isRegularFile)
                        .filter(path -> isNativeLibrary(path.getFileName().toString()))
                        .toArray(Path[]::new);
                for (Path file : files) {
                    Path destination = destination(directory, source.relativize(file).toString());
                    Files.copy(file, destination);
                    destination.toFile().deleteOnExit();
                }
            }
        } else {
            throw new IOException("Unsupported native resource protocol: " + jniResource.getProtocol());
        }
        return directory;
    }

    private static Path destination(Path directory, String relative) throws IOException {
        Path result = directory.resolve(relative).normalize();
        if (!result.startsWith(directory) || result.equals(directory)) {
            throw new IOException("Native resource escapes its extraction directory: " + relative);
        }
        Path parent = result.getParent();
        if (!Files.exists(parent)) {
            Files.createDirectories(parent);
            parent.toFile().deleteOnExit();
        }
        return result;
    }

    private static boolean isNativeLibrary(String name) {
        return name.endsWith(".dylib") || name.endsWith(".dll")
                || name.endsWith(".so") || name.contains(".so.");
    }

    public static boolean isLoaded() {
        return loaded;
    }
}
