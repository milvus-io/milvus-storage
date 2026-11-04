FROM ubuntu:22.04

# Match the Ubuntu/GCC runtime used by the build job. Conan dependencies are
# already bundled in the gtest artifact; no compilation happens in this image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libaio1 libasan6 libatomic1 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
